import os
import glob
import shutil
import json
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.cuda.amp import GradScaler, autocast # Import for AMP
import monai
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch, list_data_collate
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data.image_writer import NibabelWriter
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd, # Key transform for combining modalities
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    NormalizeIntensityd, # Alternative to ScaleIntensityRanged
    ResizeWithPadOrCropd,
)
from monai.utils import set_determinism
import numpy as np
import pandas as pd
import SimpleITK as sitk # For NIfTI I/O if needed, MONAI's LoadImaged usually handles it

# print_config()
set_determinism(seed=0) # For reproducibility


# @title Step 2 & 3: Define Paths, Hyperparameters, and Prepare/Split Data Dictionaries

import os
import random # For shuffling and subset selection

# --- ❗ CRITICAL: Define the path to your SINGLE directory of labeled BraTS data on Chameleon ---
# This is the directory that CONTAINS all your patient folders with images AND segmentation masks
# (e.g., your current 'BraTS_Data_Train' folder)
labeled_dataset_base_dir = '/home/cc/swin_unetr_train/BraTS_Data_Train/'  # <--- MODIFY THIS!
print(f"✅ Labeled dataset base directory set to: {labeled_dataset_base_dir}")

if not os.path.isdir(labeled_dataset_base_dir):
    print(f"🚨🚨🚨 ERROR: Dataset directory '{labeled_dataset_base_dir}' does not exist! Please check the path. 🚨🚨🚨")
    # Consider exiting or raising an error if path is crucial and not found
    # exit()

# 1. Output directory on Chameleon
# Example: Saving to a subfolder in your project directory on Chameleon
main_output_parent_dir = '/home/cc/swin_unetr_train/' # Parent for output folder
output_folder_name = 'SwinUNETR_Output_Patch96_Real'
output_directory = os.path.join(main_output_parent_dir, output_folder_name)  # <--- MODIFY THIS if needed
os.makedirs(output_directory, exist_ok=True)
print(f">>>> Trained models and logs will be saved to: {output_directory}")

checkpoint_dir = os.path.join(output_directory, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f">>>> Periodic checkpoints will be saved to: {checkpoint_dir}")

val_output_dir = os.path.join(output_directory, "validation_predictions")
os.makedirs(val_output_dir, exist_ok=True)
print(f">>>> Intermediate validation predictions will be saved to: {val_output_dir}")


training_log_csv_path = os.path.join(output_directory, "training_log.csv")
print(f">>>> Training metrics will be logged to: {training_log_csv_path}")

# 2. Modality and label file suffixes (Ensure these match your files)
modality_keys = ["t1c", "t1n", "t2f", "t2w"]
modality_file_suffixes = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}
label_file_suffix = "-seg.nii.gz"

# 3. Hyperparameters
IMG_SIZE = (96, 96, 96)
INPUT_CHANNELS = 4
OUTPUT_CLASSES = 4
LEARNING_RATE = 5e-5
MAX_EPOCHS = 1000  # Adjust for full run vs. test run (e.g., 5 for PoC)
BATCH_SIZE = 5    # Adjust based on GPU memory.
VAL_INTERVAL = 4  # validation after every epoch

print(f"Model parameters: Image Size={IMG_SIZE}, Input Channels={INPUT_CHANNELS}, Output Classes={OUTPUT_CLASSES}, Batch Size={BATCH_SIZE}, Max Epochs={MAX_EPOCHS}")

# --- Helper function to create data dictionaries from a base directory ---
def create_data_dicts_from_folder(base_dir, mod_suffixes, lbl_suffix, max_debug=0):
    data_list = []
    skipped = 0
    try:
        if not os.path.isdir(base_dir):
            print(f"🚨 ERROR: Provided base directory '{base_dir}' for data dict creation does not exist.")
            return [], 0
        patient_folders = sorted([os.path.join(base_dir, name) for name in os.listdir(base_dir)
                                  if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("BraTS-GLI-")])
        if not patient_folders:
            print(f"🚨 WARNING: No patient folders starting with 'BraTS-GLI-' found in '{base_dir}'.")
        else:
            print(f"Found {len(patient_folders)} patient folders in '{base_dir}'.")

        for i, patient_folder in enumerate(patient_folders):
            patient_id = os.path.basename(patient_folder)
            debug_this_patient = (i < max_debug)
            if debug_this_patient:
                print(f"\nProcessing patient {i+1}/{len(patient_folders)}: ID = {patient_id}, Folder = {patient_folder}")

            files_dict = {}
            valid_sample = True
            for key, suffix in mod_suffixes.items():
                file_path = os.path.join(patient_folder, patient_id + suffix)
                if debug_this_patient: print(f"  Checking for modality file for key '{key}': {file_path}")
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    files_dict[key] = file_path
                    if debug_this_patient: print(f"    ✅ Found file: {file_path}")
                else:
                    if debug_this_patient: print(f"    ❌ File NOT found: {file_path}. Marking invalid.")
                    valid_sample = False; break

            if not valid_sample:
                if debug_this_patient: print(f"  Skipping patient {patient_id} due to missing modality.")
                skipped += 1; continue

            label_path = os.path.join(patient_folder, patient_id + lbl_suffix)
            if debug_this_patient: print(f"  Checking for label file: {label_path}")
            if os.path.exists(label_path) and os.path.isfile(label_path):
                files_dict["label"] = label_path
                if debug_this_patient: print(f"    ✅ Label found: {label_path}")
            else:
                if debug_this_patient: print(f"    ❌ Label file NOT found: {label_path}. Skipping patient.")
                skipped += 1; continue

            data_list.append(files_dict)
            if debug_this_patient: print(f"  ✅ Patient {patient_id} successfully added.")

        print(f"✅ Created {len(data_list)} data_dicts from '{base_dir}' (skipped {skipped}).")

    except FileNotFoundError:
        print(f"🚨🚨🚨 FATAL ERROR: The directory '{base_dir}' was not found during data dict creation!")
        return [], skipped
    except Exception as e:
        print(f"🚨🚨🚨 An unexpected error occurred during data dict creation for '{base_dir}': {e}")
        return [], skipped

    return data_list, skipped

# --- Prepare ALL Labeled Data Dictionaries ---
print("\n--- Preparing All Labeled Data ---")
all_labeled_data_dicts, total_skipped = create_data_dicts_from_folder(
    labeled_dataset_base_dir,
    modality_file_suffixes,
    label_file_suffix,
    max_debug=0 # Set to >0 to debug if it's still not finding files
)

# --- Shuffle and Split ALL Labeled Data into Training, Validation, and Testing Sets ---
num_total_labeled_samples = len(all_labeled_data_dicts)

if num_total_labeled_samples > 0:
    print(f"\nTotal labeled samples found: {num_total_labeled_samples}")
    random.seed(42) # For reproducible shuffling
    random.shuffle(all_labeled_data_dicts)

    # Define split ratios
    TRAIN_RATIO = 0.70  # 70% for training
    VAL_RATIO = 0.15    # 15% for validation
    # TEST_RATIO will be 1.0 - TRAIN_RATIO - VAL_RATIO (i.e., 15%)

    num_train_split = int(num_total_labeled_samples * TRAIN_RATIO)
    num_val_split = int(num_total_labeled_samples * VAL_RATIO)

    full_train_files = all_labeled_data_dicts[:num_train_split]
    full_val_files = all_labeled_data_dicts[num_train_split : num_train_split + num_val_split]
    full_test_files = all_labeled_data_dicts[num_train_split + num_val_split :] # The rest for testing

    print(f"Split into: Full Training samples: {len(full_train_files)}, Full Validation samples: {len(full_val_files)}, Full Test samples: {len(full_test_files)}")
else:
    print("🚨 ERROR: No labeled data available to split.")
    full_train_files = []
    full_val_files = []
    full_test_files = []

# Save the list of test file dictionaries
test_files_list_path = os.path.join(output_directory, "test_set_patient_files.json")
try:
    with open(test_files_list_path, 'w') as f:
        json.dump(full_test_files, f, indent=4)
    print(f">>>> Test set file list (containing {len(full_test_files)} samples) saved to: {test_files_list_path}")
except Exception as e:
    print(f"🚨 ERROR saving test set file list: {e}")


# For training, you'll use full_train_files and full_val_files (or their subsets)
train_files = full_train_files # Or full_train_files[:subset_size_train]
val_files = full_val_files     # Or full_val_files[:subset_size_val]

# The 'full_test_files' list should be saved and only used after all training and tuning is complete.
# You might want to save this list of file dictionaries to disk.
# For example:
# import json
# test_files_save_path = os.path.join(output_directory, "test_set_file_list.json")
# with open(test_files_save_path, 'w') as f:
#     json.dump(full_test_files, f, indent=4)
# print(f"Test set file list saved to: {test_files_save_path}")


# --- (Optional) Select a Percentage Subset from the *newly split* Train/Val for PoC/Debugging ---
SUBSET_PERCENTAGE_TRAIN = 1  # Use 10% of the *new* training data for quick test
SUBSET_PERCENTAGE_VAL = 1    # Use 10% of the *new* validation data for quick test
# Set to 1.0 to use all data from the new train/val splits

if len(full_train_files) > 0:
    if SUBSET_PERCENTAGE_TRAIN < 1.0 :
        num_train_subset = int(len(full_train_files) * SUBSET_PERCENTAGE_TRAIN)
        # No need to shuffle again if full_train_files is already a result of a shuffle
        train_files = full_train_files[:num_train_subset]
        print(f"\nUsing a {SUBSET_PERCENTAGE_TRAIN*100:.0f}% subset of actual training data: {len(train_files)} samples.")
    else:
        train_files = full_train_files
        print(f"\nUsing all actual training data: {len(train_files)} samples.")
else:
    train_files = []

if len(full_val_files) > 0:
    if SUBSET_PERCENTAGE_VAL < 1.0:
        num_val_subset = int(len(full_val_files) * SUBSET_PERCENTAGE_VAL)
        # No need to shuffle again
        val_files = full_val_files[:num_val_subset]
        print(f"Using a {SUBSET_PERCENTAGE_VAL*100:.0f}% subset of actual validation data: {len(val_files)} samples.")
    else:
        val_files = full_val_files
        print(f"Using all actual validation data: {len(val_files)} samples.")

#
# if train_files:
#     print(f"First training sample dict to be used: {train_files[0]}")
# if val_files:
#     print(f"First validation sample dict to be used: {val_files[0]}")
# else:
#     print("⚠️ Warning: No validation files selected. Validation DataLoader will be empty.")




# @title Step 4: Define Data Transforms

# Keys for LoadImaged based on what we put in data_dicts
# The 'image' key will be created by ConcatItemsd
image_loading_keys = modality_keys
all_keys_to_load = modality_keys + ["label"]

# Target spacing for resampling
TARGET_SPACING = (1.0, 1.0, 1.0) # Typical for BraTS

train_transforms = Compose(
    [
        LoadImaged(keys=all_keys_to_load, image_only=False), # image_only=False to keep metadata for spacing
        EnsureChannelFirstd(keys=all_keys_to_load),
        EnsureTyped(keys=all_keys_to_load, dtype=torch.float32), # Ensure data is float
        Orientationd(keys=all_keys_to_load, axcodes="RAS"),
        Spacingd(
            keys=all_keys_to_load, # Apply to all loaded images and label
            pixdim=TARGET_SPACING,
            mode=["bilinear"] * len(image_loading_keys) + ["nearest"], # Bilinear for images, nearest for label
        ),
        # --- Intensity Preprocessing for each modality before concatenation ---
        # Example: ScaleIntensityRanged for each modality
        # You might need to find appropriate a_min, a_max for your data (e.g., 0.5 to 99.5 percentile)
        # These are just illustrative values
        ScaleIntensityRanged(keys=["t1c"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t1n"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2f"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True), # FLAIR might have different range
        ScaleIntensityRanged(keys=["t2w"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        # Or use NormalizeIntensityd(keys=image_loading_keys, nonzero=True, channel_wise=True), # Z-score normalization for each channel

        # --- Concatenate modalities into a single 'image' ---
        ConcatItemsd(keys=image_loading_keys, name="image", dim=0), # dim=0 for channel-first

        # --- Transforms on the combined 4-channel 'image' and 'label' ---
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMG_SIZE,
            pos=1,
            neg=1,
            num_samples=1, # Number of patches per image
            image_key="image",
            image_threshold=0, # Threshold for considering a voxel as foreground in the image
        ),
        # --- Augmentations ---
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)), # Rotate in xy plane
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=all_keys_to_load, image_only=False),
        EnsureChannelFirstd(keys=all_keys_to_load),
        EnsureTyped(keys=all_keys_to_load, dtype=torch.float32),
        Orientationd(keys=all_keys_to_load, axcodes="RAS"),
        Spacingd(
            keys=all_keys_to_load,
            pixdim=TARGET_SPACING,
            mode=["bilinear"] * len(image_loading_keys) + ["nearest"], # Bilinear for images, nearest for label
        ),
        ScaleIntensityRanged(keys=["t1c"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t1n"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2f"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2w"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        # Or use NormalizeIntensityd(keys=image_loading_keys, nonzero=True, channel_wise=True),

        ConcatItemsd(keys=image_loading_keys, name="image", dim=0),
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]]), # Ensure consistent cropping
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)

# @title Step 5: Create Datasets and DataLoaders (Revised for Colab Memory)

print("Attempting to create Datasets and DataLoaders...")
print(f"Number of training files (dictionaries): {len(train_files)}")
print(f"Number of validation files (dictionaries): {len(val_files)}")

# Use monai.data.Dataset (RECOMMENDED for Colab to save RAM)
# This loads and transforms data on-the-fly, using much less RAM.
print("Using monai.data.Dataset (loads data on-the-fly)...")
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

# --- DataLoader Settings for Reduced Memory Usage ---
# num_workers=0 means data is loaded in the main process.
num_dataloader_workers = 16 # Try 2, then 1, then 0 if crashing persists
print(f"Using num_workers={num_dataloader_workers} for DataLoaders.")

# Set pin_memory=False to save pinned RAM.
use_pin_memory = True
print(f"Using pin_memory={use_pin_memory} for DataLoaders.")

# Use list_data_collate for MONAI Datasets that return a list of dictionaries
# (common with RandCropByPosNegLabeld if num_samples > 1 during training)
train_loader = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=num_dataloader_workers,
                          pin_memory=use_pin_memory,
                          collate_fn=list_data_collate, # Important for RandCropByPosNegLabeld
                          persistent_workers= (num_dataloader_workers > 0) # Can help speed up if num_workers > 0
                         )

val_loader = DataLoader(val_ds,
                        batch_size=1, # Keep validation batch_size to 1 for whole volume inference usually
                        shuffle=False,
                        num_workers=num_dataloader_workers, # Can also be 0 for validation
                        pin_memory=use_pin_memory
                       )

print("✅ Datasets and DataLoaders created successfully!")
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")



# @title Step 6: Define Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Swin UNETR model
# For BraTS, output classes might be 3 (ET, TC, WT) or 4 (BG, ET, TC, WT / or BG, NCR, ED, ET)
# This depends on how your segmentation labels are defined and mapped.
# Assuming OUTPUT_CLASSES = 3 for 3 foreground classes.
# If your labels are (0:BG, 1:C1, 2:C2, 3:C3), then out_channels = 4
# Let's assume model output logits for 3 foreground classes, and background is implicit or handled by loss.
# If DiceCELoss has softmax=True and to_onehot_y=True, it can handle integer labels [0, K-1] for K classes.
# If your labels are 1,2,4 for ET,TC,WT, you need to remap them (e.g. in LoadImaged or a custom transform)
# to 0,1,2 or 1,2,3 if not using background class explicitly in model output channels.
# For OUTPUT_CLASSES=3 (e.g. predicting ET, TC, WT directly as channels 0,1,2)
# and labels are 1,2,3 for these.

model = SwinUNETR(
    # img_size=IMG_SIZE,
    in_channels=INPUT_CHANNELS, # 4 modalities
    out_channels=OUTPUT_CLASSES, # Number of segmentation classes
    feature_size=48,         # Default is 48, can be adjusted
    use_checkpoint=True,     # Gradient checkpointing for memory saving
).to(device)

# Loss function
# DiceCELoss is common for BraTS.
# to_onehot_y=True if your labels are class indices (e.g., 0, 1, 2, 3)
# softmax=True applies softmax to model output logits
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background= (OUTPUT_CLASSES==4) ) # Adjust include_background based on OUTPUT_CLASSES

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# Initialize GradScaler for AMP
scaler = GradScaler()

# --- Metrics for validation ---
# We will use four separate metric objects to track Mean, ET, TC, and WT Dice scores.
# For BraTS, the evaluation is typically on foreground classes, so include_background=False is standard.
# However, since we create binary masks for ET/TC/WT, we compare foreground (1) vs background (0),
# so include_background=False is functionally correct here as it will compute Dice for the foreground=1 class.
dice_metric_mean = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_et = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_tc = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_wt = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Post-processing for metric calculation
# This converts model output logits to a one-hot format, e.g., (B, C, H, W, D)
# where C is the number of classes.
post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=OUTPUT_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=OUTPUT_CLASSES)])
post_pred_save = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=None, n_classes=None)])


# @title Step 7: Training Loop (with AMP and Validation Saving)

PATIENCE_CHECKS = 20
patience_counter = 0
best_val_metric_for_early_stopping = -1.0
early_stop_triggered = False

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []

# Create separate lists to store each validation metric over time for plotting
metric_values_mean = []
metric_values_et = []
metric_values_tc = []
metric_values_wt = []

print(f"Early stopping patience set to {PATIENCE_CHECKS} validation checks.")

CONTINUE_TRAINING = False
start_epoch = 0
# SAVE_CHECKPOINT_FREQ = 40

# --- Initialize or Load Log File ---
log_columns = ["epoch", "avg_train_loss", "val_mean_dice", "val_dice_et", "val_dice_tc", "val_dice_wt"]
log_df = None
training_log_csv_path = os.path.join(output_directory, "training_log.csv")

if os.path.exists(training_log_csv_path) and CONTINUE_TRAINING:
    print(f"Loading existing training log from: {training_log_csv_path}")
    try:
        log_df = pd.read_csv(training_log_csv_path)
    except Exception as e:
        print(f"Error loading log file: {e}. Starting new log.")
        log_df = pd.DataFrame(columns=log_columns)
else:
    print("Starting new training log.")
    log_df = pd.DataFrame(columns=log_columns)

for epoch in range(start_epoch, MAX_EPOCHS):
    print("-" * 20)
    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    epoch_train_start_time = time.time()
    model.train()
    epoch_loss = 0
    step = 0
    for i, batch_data in enumerate(train_loader):
        step += 1
        try:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(f"Step {step}/{len(train_loader)}, Training_loss: {loss.item():.4f}")
        except Exception as e:
            print(f"🚨🚨 FAILED at Step {step}. Error: {e}")
            raise e

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_train_duration = time.time() - epoch_train_start_time
    print(f"Epoch {epoch + 1} Average Training Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch + 1} Training Time: {epoch_train_duration:.2f} seconds")

    scheduler.step()

    # Initialize current epoch's log data with NaN for metric columns
    current_epoch_log = {"epoch": epoch + 1, "avg_train_loss": epoch_loss, "val_mean_dice": np.nan, "val_dice_et": np.nan, "val_dice_tc": np.nan, "val_dice_wt": np.nan}

    if (epoch + 1) % VAL_INTERVAL == 0:
        epoch_val_start_time = time.time()
        model.eval()

        ### --- NEW CODE START: Setup for saving validation predictions --- ###
        # Create a subdirectory for this specific epoch's validation outputs
        # current_epoch_val_dir = os.path.join(val_output_dir, f"epoch_{epoch + 1:04d}")
        # os.makedirs(current_epoch_val_dir, exist_ok=True)

        ### --- NEW CODE END --- ###

        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                with autocast():
                    val_outputs = sliding_window_inference(val_inputs, roi_size=IMG_SIZE, sw_batch_size=BATCH_SIZE, predictor=model, overlap=0.5)

                # Process for metric calculation
                val_outputs_one_hot = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_one_hot = [post_label(i) for i in decollate_batch(val_labels)]

                # --- vvvvv 诊断代码：检查用于计算指标的张量 vvvvv ---
                if i == 0: # 只检查每个验证轮次的第一个样本，避免信息刷屏
                    print("\n--- METRIC CALCULATION DIAGNOSTICS (EPOCH {}) ---".format(epoch + 1))
                    # 检查预测的one-hot张量
                    pred_tensor_for_metric = val_outputs_one_hot[0]
                    print(f"Shape of Prediction Tensor for Metric: {pred_tensor_for_metric.shape}")
                    print(f"Unique values in Prediction Tensor: {torch.unique(pred_tensor_for_metric)}")

                    # 检查标签的one-hot张量
                    label_tensor_for_metric = val_labels_one_hot[0]
                    print(f"Shape of Label Tensor for Metric: {label_tensor_for_metric.shape}")
                    print(f"Unique values in Label Tensor: {torch.unique(label_tensor_for_metric)}")
                    print("--------------------------------------------------\n")
                # --- ^^^^^ 诊断代码结束 ^^^^^ ---


                # Your existing metric calculation logic...
                output_tensor = val_outputs_one_hot[0]
                label_tensor = val_labels_one_hot[0]

                et_pred_mask = output_tensor[3:4]
                et_label_mask = label_tensor[3:4]

                # Tumor Core (TC) is channels 1 (NCR/NET) + 3 (ET)
                tc_pred_mask = output_tensor[1:2].bool() | output_tensor[3:4].bool()
                tc_label_mask = label_tensor[1:2].bool() | label_tensor[3:4].bool()

                # Whole Tumor (WT) is channels 1 (NCR/NET) + 2 (ED) + 3 (ET)
                wt_pred_mask = output_tensor[1:2].bool() | output_tensor[2:3].bool() | output_tensor[3:4].bool()
                wt_label_mask = label_tensor[1:2].bool() | label_tensor[2:3].bool() | label_tensor[3:4].bool()

                dice_metric_mean(y_pred=val_outputs_one_hot, y=val_labels_one_hot)
                dice_metric_et(y_pred=[et_pred_mask], y=[et_label_mask])
                dice_metric_tc(y_pred=[tc_pred_mask], y=[tc_label_mask])
                dice_metric_wt(y_pred=[wt_pred_mask], y=[wt_label_mask])

                # ### --- NEW CODE START: Saving the prediction file --- ###
                # # Get patient ID for the filename
                # try:
                #     # Assumes the first modality's filename is representative
                #     patient_id = os.path.basename(val_data["image_meta_dict"]["filename_or_obj"][0]).split(modality_file_suffixes[modality_keys[0]])[0]
                # except Exception:
                #     patient_id = f"val_sample_{i:03d}"
                #
                # print(f"  Validating and saving prediction for: {patient_id}")
                #
                # # --- FIX: Decollate the batch to get metadata for the single sample ---
                # # This removes the extra "batch" dimension from the metadata dictionary.
                # # We take the first element [0] because the validation batch size is 1.
                # val_data_decollated = decollate_batch(val_data)[0]
                #
                #
                # # Use the dedicated post-processing transform for saving
                # pred_to_save = post_pred_save(decollate_batch(val_outputs)[0])
                #
                # # --- FINAL, MOST PRECISE DIAGNOSTIC ---
                # print("\nDEBUG: Starting final diagnostic for GPU-to-CPU transfer...")
                #
                # # 1. 我们知道 pred_to_save 在GPU上是正确的
                # # pred_to_save = post_pred_save(...)
                #
                # # 2. 显式地将数据从GPU移动到CPU
                # print("  Step 1: Moving tensor to CPU...")
                # pred_cpu = pred_to_save.cpu()
                #
                # # 3. 强制CUDA同步，确保所有GPU操作（包括拷贝）已完成
                # #    这是一个关键的调试步骤
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # print("  Step 2: CUDA synchronized.")
                #
                # # 4. 关键检查：检查刚刚移动到CPU上的张量的内容
                # cpu_unique_vals = torch.unique(pred_cpu)
                # print(f"  Step 3: CRITICAL CHECK - Unique values on CPU tensor: {cpu_unique_vals}")
                #
                # # 5. 将CPU张量转换为NumPy数组，并再次检查
                # print("  Step 4: Converting CPU tensor to NumPy array...")
                # final_numpy_array = pred_cpu.numpy().astype(np.uint8)
                # numpy_unique_vals = np.unique(final_numpy_array)
                # print(f"  Step 5: CRITICAL CHECK - Unique values in final NumPy array: {numpy_unique_vals}")
                #
                # # 6. 最后，使用我们已知能工作的 SimpleITK 再次尝试保存
                # if numpy_unique_vals.size > 1 or (numpy_unique_vals.size == 1 and numpy_unique_vals[0] != 0):
                #     print("  Step 6: Data appears valid, attempting to save with SimpleITK...")
                #     try:
                #         sitk_image_pred = sitk.GetImageFromArray(final_numpy_array)
                #         output_filename = os.path.join(current_epoch_val_dir, f"{patient_id}_FINAL_TEST.nii.gz")
                #         sitk.WriteImage(sitk_image_pred, output_filename)
                #         print(f"    ✅ FINAL TEST file saved successfully.")
                #     except Exception as e:
                #         print(f"    ❌ FAILED during final SimpleITK save: {e}")
                # else:
                #     print("  Step 6: Skipping save because NumPy array is empty or all zeros.")
                # print("--- End of final diagnostic ---\n")

            # Aggregate all metrics after iterating through the validation set
            current_mean_dice = dice_metric_mean.aggregate().item()
            current_et_dice = dice_metric_et.aggregate().item()
            current_tc_dice = dice_metric_tc.aggregate().item()
            current_wt_dice = dice_metric_wt.aggregate().item()

            # Reset all metrics for the next validation run
            dice_metric_mean.reset()
            dice_metric_et.reset()
            dice_metric_tc.reset()
            dice_metric_wt.reset()

            # Append new scores to our lists for plotting
            metric_values_mean.append(current_mean_dice)
            metric_values_et.append(current_et_dice)
            metric_values_tc.append(current_tc_dice)
            metric_values_wt.append(current_wt_dice)

            if current_mean_dice > best_metric:
                best_metric = current_mean_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_directory, "best_metric_model_swinunetr.pth"))
                print(f"Saved new best metric model (Epoch {best_metric_epoch}, Mean Dice: {best_metric:.4f})")

            # Early Stopping Logic
            if current_mean_dice > best_val_metric_for_early_stopping:
                best_val_metric_for_early_stopping = current_mean_dice
                patience_counter = 0
                print(f"EarlyStopping: Validation metric improved to {best_val_metric_for_early_stopping:.4f}. Patience reset.")
            else:
                patience_counter += 1
                print(f"EarlyStopping: Validation metric did not improve for {patience_counter} validation check(s) out of {PATIENCE_CHECKS}.")

            if patience_counter >= PATIENCE_CHECKS:
                print(f"EarlyStopping: Patience limit ({PATIENCE_CHECKS} checks) reached. Setting stop trigger.")
                early_stop_triggered = True

            # Update the log dictionary for this validation epoch
            current_epoch_log["val_mean_dice"] = current_mean_dice
            current_epoch_log["val_dice_et"] = current_et_dice
            current_epoch_log["val_dice_tc"] = current_tc_dice
            current_epoch_log["val_dice_wt"] = current_wt_dice

            print(f"Validation - Epoch: {epoch + 1}")
            print(f"  Mean Dice: {current_mean_dice:.4f}")
            print(f"  ET Dice:   {current_et_dice:.4f}")
            print(f"  TC Dice:   {current_tc_dice:.4f}")
            print(f"  WT Dice:   {current_wt_dice:.4f}")
            print(f"Best mean Dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Append the log for the current epoch to the DataFrame and save to CSV
    new_log_entry_df = pd.DataFrame([current_epoch_log])
    log_df = pd.concat([log_df, new_log_entry_df], ignore_index=True)
    try:
        log_df.to_csv(training_log_csv_path, index=False)
    except Exception as e:
        print(f"Error saving training log: {e}")

    if early_stop_triggered:
        print(f"Terminating training early at epoch {epoch + 1} due to early stopping.")
        break # Break from the main training epoch loop


# @title Step 8: Save Final Model
torch.save(model.state_dict(), os.path.join(output_directory, "final_model_swinunetr.pth"))
print(f"Training completed. Best Mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}.")
print(f"Models saved in: {output_directory}")


# @title Step 9: Plot Loss and Metrics (3 Subplots)

import matplotlib.pyplot as plt

# Create a figure with a 1x3 grid layout. Increase figsize width to accommodate the extra plot.
plt.figure("train", (24, 6))

# --- Plot 1: Training Loss ---
plt.subplot(1, 3, 1)
plt.title("Epoch Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# The x-axis for training loss is every epoch from 1 to N
train_epochs = range(1, len(epoch_loss_values) + 1)
plt.plot(train_epochs, epoch_loss_values, color='red', label="Train Loss")
plt.grid(True)
plt.legend()

# --- Plot 2: Individual Validation Dice Scores (ET, TC, WT) ---
plt.subplot(1, 3, 2)
plt.title("Validation Dice Scores (ET, TC, WT)")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
# X-axis for validation metrics should correspond to validation epochs
val_epochs = [(i + 1) * VAL_INTERVAL for i in range(len(metric_values_et))] # Use any of the metric lists for length
plt.plot(val_epochs, metric_values_et, color='red', marker='s', linestyle='--', label="ET Dice")
plt.plot(val_epochs, metric_values_tc, color='green', marker='^', linestyle='-.', label="TC Dice")
plt.plot(val_epochs, metric_values_wt, color='purple', marker='d', linestyle=':', label="WT Dice")
plt.grid(True)
plt.legend() # Add a legend to distinguish the lines
plt.ylim([0, 1]) # Set y-axis limits for Dice score from 0 to 1

# --- Plot 3: Overall Mean Dice Score ---
plt.subplot(1, 3, 3)
plt.title("Validation Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(val_epochs, metric_values_mean, color='blue', marker='o', linestyle='-', label="Mean Dice")
plt.grid(True)
plt.legend()
plt.ylim([0, 1]) # Also set y-axis limits here for consistency

# --- Final Adjustments and Saving ---
plt.suptitle("Training and Validation Curves", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig(os.path.join(output_directory, "training_curves_swinunetr.png"))
# plt.show() # Keep commented out for server runs
print("✅ Training curves plot with 3 subplots saved.")

