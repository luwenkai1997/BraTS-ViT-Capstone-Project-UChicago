import os
import glob
import shutil
import json
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.cuda.amp import GradScaler, autocast
import monai
from monai.config import print_config
from monai.data.image_writer import NibabelWriter
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)
from monai.utils import set_determinism
import numpy as np
import pandas as pd
import random # For shuffling and subset selection

# print_config()
set_determinism(seed=0) # For reproducibility

# --- ❗ CRITICAL: Define Paths ---
# Path to your SINGLE directory of REAL labeled BraTS data on Chameleon
# This should be the same as 'labeled_dataset_base_dir' in real_train_swin_unetr_monai.py
real_labeled_dataset_base_dir = '/home/cc/swin_unetr_train/BraTS_Data_Train/' #
print(f"✅ Real labeled dataset base directory set to: {real_labeled_dataset_base_dir}")

if not os.path.isdir(real_labeled_dataset_base_dir):
    print(f"🚨🚨🚨 ERROR: Real dataset directory '{real_labeled_dataset_base_dir}' does not exist! Please check the path. 🚨🚨🚨")
    # Consider exiting or raising an error if path is crucial and not found
    # exit()

# --- Path to the PRETRAINED MODEL (from synthetic data training) ---
# This path should point to the output directory of your synthetic training,
# and specifically to the 'best_metric_model_swinunetr.pth' or 'final_model_swinunetr.pth'
# from that synthetic training run.
pretrained_model_path = '/home/cc/swin_unetr_train/SwinUNETR_Output_Patch96_Syn/best_metric_model_swinunetr.pth' # <--- MODIFY THIS!
print(f"✅ Pretrained model path set to: {pretrained_model_path}")

if not os.path.exists(pretrained_model_path):
    print(f"🚨🚨🚨 ERROR: Pretrained model '{pretrained_model_path}' not found! Please check the path. 🚨🚨🚨")
    # exit()

# 1. Output directory for FINE-TUNING results
main_output_parent_dir = '/home/cc/swin_unetr_train/'
output_folder_name = 'SwinUNETR_Output_Patch96_Finetune' # New output folder for finetuning
output_directory = os.path.join(main_output_parent_dir, output_folder_name)
os.makedirs(output_directory, exist_ok=True)
print(f">>>> Trained models and logs for finetuning will be saved to: {output_directory}")

checkpoint_dir = os.path.join(output_directory, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f">>>> Periodic checkpoints will be saved to: {checkpoint_dir}")

val_output_dir = os.path.join(output_directory, "validation_predictions")
os.makedirs(val_output_dir, exist_ok=True)
print(f">>>> Intermediate validation predictions will be saved to: {val_output_dir}")

training_log_csv_path = os.path.join(output_directory, "finetune_training_log.csv") # Separate log for finetuning
print(f">>>> Training metrics will be logged to: {training_log_csv_path}")

# 2. Modality and label file suffixes (Ensure these match your real data files)
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
OUTPUT_CLASSES = 4 # BraTS typically uses 4 classes (0: Background, 1: NCR/NET, 2: ED, 3: ET)
LEARNING_RATE = 1e-6 # ❗ IMPORTANT: Use a smaller learning rate for fine-tuning
MAX_EPOCHS = 300 # ❗ IMPORTANT: Fine-tuning usually requires fewer epochs than pre-training
BATCH_SIZE = 5 # Adjust based on GPU memory. Keep consistent with pre-training if possible.
VAL_INTERVAL = 4 # validation after every epoch

print(f"Model parameters for finetuning: Image Size={IMG_SIZE}, Input Channels={INPUT_CHANNELS}, Output Classes={OUTPUT_CLASSES}, Batch Size={BATCH_SIZE}, Max Epochs={MAX_EPOCHS}, Learning Rate={LEARNING_RATE}")

# --- Helper function to create data dictionaries from a base directory (same as in original scripts) ---
def create_data_dicts_from_folder(base_dir, mod_suffixes, lbl_suffix, max_debug=0): #
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

# --- Prepare ALL Real Labeled Data Dictionaries ---
print("\n--- Preparing All Real Labeled Data for Fine-tuning ---")
all_labeled_data_dicts, total_skipped = create_data_dicts_from_folder(
    real_labeled_dataset_base_dir,
    modality_file_suffixes,
    label_file_suffix,
    max_debug=0
)

# --- Shuffle and Split ALL Real Labeled Data into Training, Validation, and Testing Sets ---
num_total_labeled_samples = len(all_labeled_data_dicts)

if num_total_labeled_samples > 0:
    print(f"\nTotal real labeled samples found: {num_total_labeled_samples}")
    random.seed(42) # For reproducible shuffling
    random.shuffle(all_labeled_data_dicts)

    # Define split ratios (same as real_train_swin_unetr_monai.py)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    num_train_split = int(num_total_labeled_samples * TRAIN_RATIO)
    num_val_split = int(num_total_labeled_samples * VAL_RATIO)

    full_train_files = all_labeled_data_dicts[:num_train_split]
    full_val_files = all_labeled_data_dicts[num_train_split : num_train_split + num_val_split]
    full_test_files = all_labeled_data_dicts[num_train_split + num_val_split :] # The rest for testing

    print(f"Split into: Full Training samples: {len(full_train_files)}, Full Validation samples: {len(full_val_files)}, Full Test samples: {len(full_test_files)}")
else:
    print("🚨 ERROR: No real labeled data available to split.")
    full_train_files = []
    full_val_files = []
    full_test_files = []

# Save the list of test file dictionaries (important to keep this consistent)
test_files_list_path = os.path.join(output_directory, "test_set_patient_files_finetune.json")
try:
    with open(test_files_list_path, 'w') as f:
        json.dump(full_test_files, f, indent=4)
    print(f">>>> Test set file list (containing {len(full_test_files)} samples) saved to: {test_files_list_path}")
except Exception as e:
    print(f"🚨 ERROR saving test set file list: {e}")

# For training, you'll use full_train_files and full_val_files (or their subsets)
train_files = full_train_files # Using all real training data for finetuning
val_files = full_val_files # Using all real validation data for finetuning

# --- (Optional) Select a Percentage Subset from the *newly split* Train/Val for PoC/Debugging ---
# For fine-tuning, you often want to use all available real data.
SUBSET_PERCENTAGE_TRAIN = 1.0
SUBSET_PERCENTAGE_VAL = 1.0

if len(full_train_files) > 0:
    if SUBSET_PERCENTAGE_TRAIN < 1.0 :
        num_train_subset = int(len(full_train_files) * SUBSET_PERCENTAGE_TRAIN)
        train_files = full_train_files[:num_train_subset]
        print(f"\nUsing a {SUBSET_PERCENTAGE_TRAIN*100:.0f}% subset of actual training data for finetuning: {len(train_files)} samples.")
    else:
        train_files = full_train_files
        print(f"\nUsing all actual training data for finetuning: {len(train_files)} samples.")
else:
    train_files = []

if len(full_val_files) > 0:
    if SUBSET_PERCENTAGE_VAL < 1.0:
        num_val_subset = int(len(full_val_files) * SUBSET_PERCENTAGE_VAL)
        val_files = full_val_files[:num_val_subset]
        print(f"Using a {SUBSET_PERCENTAGE_VAL*100:.0f}% subset of actual validation data for finetuning: {len(val_files)} samples.")
    else:
        val_files = full_val_files
        print(f"Using all actual validation data for finetuning: {len(val_files)} samples.")

if train_files:
    print(f"First training sample dict to be used: {train_files[0]}")
if val_files:
    print(f"First validation sample dict to be used: {val_files[0]}")
else:
    print("⚠️ Warning: No validation files selected. Validation DataLoader will be empty.")


# @title Step 4: Define Data Transforms (Same as real_train, ensure consistency)

# Keys for LoadImaged based on what we put in data_dicts
image_loading_keys = modality_keys
all_keys_to_load = modality_keys + ["label"]

# Target spacing for resampling
TARGET_SPACING = (1.0, 1.0, 1.0)

train_transforms = Compose(
    [
        LoadImaged(keys=all_keys_to_load, image_only=False),
        EnsureChannelFirstd(keys=all_keys_to_load),
        EnsureTyped(keys=all_keys_to_load, dtype=torch.float32),
        Orientationd(keys=all_keys_to_load, axcodes="RAS"),
        Spacingd(
            keys=all_keys_to_load,
            pixdim=TARGET_SPACING,
            mode=["bilinear"] * len(image_loading_keys) + ["nearest"],
        ),
        ScaleIntensityRanged(keys=["t1c"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t1n"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2f"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2w"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=image_loading_keys, name="image", dim=0),
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMG_SIZE,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
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
            mode=["bilinear"] * len(image_loading_keys) + ["nearest"],
        ),
        ScaleIntensityRanged(keys=["t1c"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t1n"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2f"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["t2w"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=image_loading_keys, name="image", dim=0),
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)

# @title Step 5: Create Datasets and DataLoaders (Revised for Colab Memory)

print("Attempting to create Datasets and DataLoaders for finetuning...")
print(f"Number of training files (dictionaries): {len(train_files)}")
print(f"Number of validation files (dictionaries): {len(val_files)}")

print("Using monai.data.Dataset (loads data on-the-fly)...")
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

num_dataloader_workers = 16 # Keep consistent or adjust based on system
print(f"Using num_workers={num_dataloader_workers} for DataLoaders.")

use_pin_memory = True # Keep consistent
print(f"Using pin_memory={use_pin_memory} for DataLoaders.")

train_loader = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=num_dataloader_workers,
                          pin_memory=use_pin_memory,
                          collate_fn=list_data_collate,
                          persistent_workers= (num_dataloader_workers > 0)
                          )

val_loader = DataLoader(val_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=num_dataloader_workers,
                        pin_memory=use_pin_memory
                        )

print("✅ Datasets and DataLoaders created successfully for finetuning!")
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")


# @title Step 6: Define Model, Loss, Optimizer (with pretrained weights loading)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SwinUNETR(
    in_channels=INPUT_CHANNELS,
    out_channels=OUTPUT_CLASSES,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# --- 核心微调步骤：加载预训练模型权重 ---
print(f"Loading pretrained model weights from: {pretrained_model_path}")
try:
    # It's generally safer to load the state_dict directly if the model architecture is identical.
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("✅ Successfully loaded pretrained weights!")
except Exception as e:
    print(f"🚨🚨 ERROR loading pretrained model weights: {e}")
    print("Proceeding with randomly initialized weights for fine-tuning. This is likely not desired.")
    # You might want to exit here if loading pretrained weights is critical

# Loss function (same as before)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background= (OUTPUT_CLASSES==4) )

# Optimizer (using the new, smaller LEARNING_RATE for finetuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Scheduler (adjust T_max based on new MAX_EPOCHS)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# Initialize GradScaler for AMP
scaler = GradScaler()

# --- Metrics for validation (same as before) ---
dice_metric_mean = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_et = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_tc = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_wt = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=OUTPUT_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=OUTPUT_CLASSES)])
post_pred_save = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=None, n_classes=None)])


# @title Step 7: Fine-tuning Loop (with AMP and Validation Saving)

PATIENCE_CHECKS = 10 # Can be adjusted for finetuning
patience_counter = 0
best_val_metric_for_early_stopping = -1.0
early_stop_triggered = False

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []

metric_values_mean = []
metric_values_et = []
metric_values_tc = []
metric_values_wt = []

print(f"Early stopping patience set to {PATIENCE_CHECKS} validation checks for fine-tuning.")

# Set to False to start fresh finetuning, or True to resume an interrupted finetuning run
CONTINUE_TRAINING_FINETUNE = False
start_epoch = 0
SAVE_CHECKPOINT_FREQ = 200 # Save checkpoints more frequently during finetuning

# --- Initialize or Load Log File for Finetuning ---
log_columns = ["epoch", "avg_train_loss", "val_mean_dice", "val_dice_et", "val_dice_tc", "val_dice_wt"]
log_df = None

if os.path.exists(training_log_csv_path) and CONTINUE_TRAINING_FINETUNE:
    print(f"Loading existing fine-tuning log from: {training_log_csv_path}")
    try:
        log_df = pd.read_csv(training_log_csv_path)
        # If loading, set start_epoch to continue from where it left off
        if not log_df.empty:
            start_epoch = log_df["epoch"].max()
            print(f"Resuming fine-tuning from epoch {start_epoch + 1}.")
            # Load the last checkpoint if resuming
            last_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pth")
            if os.path.exists(last_checkpoint):
                print(f"Loading checkpoint for epoch {start_epoch} to resume training.")
                checkpoint = torch.load(last_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                best_metric = checkpoint['best_metric']
                best_metric_epoch = checkpoint['best_metric_epoch']
                patience_counter = checkpoint['patience_counter']
                best_val_metric_for_early_stopping = checkpoint['best_val_metric_for_early_stopping']
                epoch_loss_values = checkpoint['epoch_loss_values']
                metric_values_mean = checkpoint['metric_values_mean']
                metric_values_et = checkpoint['metric_values_et']
                metric_values_tc = checkpoint['metric_values_tc']
                metric_values_wt = checkpoint['metric_values_wt']
            else:
                print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting fine-tuning from scratch with loaded pretrained model.")
                start_epoch = 0 # Reset if checkpoint not found
        else:
            print("Existing log file is empty. Starting new fine-tuning log.")
            log_df = pd.DataFrame(columns=log_columns)
    except Exception as e:
        print(f"Error loading log file: {e}. Starting new fine-tuning log.")
        log_df = pd.DataFrame(columns=log_columns)
else:
    print("Starting new fine-tuning log.")
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

    current_epoch_log = {"epoch": epoch + 1, "avg_train_loss": epoch_loss, "val_mean_dice": np.nan, "val_dice_et": np.nan, "val_dice_tc": np.nan, "val_dice_wt": np.nan}

    if (epoch + 1) % VAL_INTERVAL == 0:
        epoch_val_start_time = time.time()
        model.eval()

        # Create a subdirectory for this specific epoch's validation outputs
        current_epoch_val_dir = os.path.join(val_output_dir, f"epoch_{epoch + 1:04d}")
        os.makedirs(current_epoch_val_dir, exist_ok=True)

        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                with autocast():
                    val_outputs = sliding_window_inference(val_inputs, roi_size=IMG_SIZE, sw_batch_size=BATCH_SIZE, predictor=model, overlap=0.5)

                val_outputs_one_hot = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_one_hot = [post_label(i) for i in decollate_batch(val_labels)]

                # Metric calculation (same as real_train_swin_unetr_monai.py)
                output_tensor = val_outputs_one_hot[0]
                label_tensor = val_labels_one_hot[0]

                et_pred_mask = output_tensor[3:4]
                et_label_mask = label_tensor[3:4]

                tc_pred_mask = output_tensor[1:2].bool() | output_tensor[3:4].bool()
                tc_label_mask = label_tensor[1:2].bool() | label_tensor[3:4].bool()

                wt_pred_mask = output_tensor[1:2].bool() | output_tensor[2:3].bool() | output_tensor[3:4].bool()
                wt_label_mask = label_tensor[1:2].bool() | label_tensor[2:3].bool() | label_tensor[3:4].bool()

                dice_metric_mean(y_pred=val_outputs_one_hot, y=val_labels_one_hot)
                dice_metric_et(y_pred=[et_pred_mask], y=[et_label_mask])
                dice_metric_tc(y_pred=[tc_pred_mask], y=[tc_label_mask])
                dice_metric_wt(y_pred=[wt_pred_mask], y=[wt_label_mask])

                # Saving the prediction file
                try:
                    patient_id = os.path.basename(val_data["image_meta_dict"]["filename_or_obj"][0]).split(modality_file_suffixes[modality_keys[0]])[0]
                except Exception:
                    patient_id = f"val_sample_{i:03d}"

                # print(f"  Validating and saving prediction for: {patient_id}")

                val_data_decollated = decollate_batch(val_data)[0]
                pred_to_save = post_pred_save(decollate_batch(val_outputs)[0])

                # Using NibabelWriter as in MONAI examples for saving NIfTI files
                # Need to use NibabelWriter directly if using MONAI's default transforms on metadata
                writer = NibabelWriter(output_dir=current_epoch_val_dir, squeeze_image=True, dtype=np.uint8)
                writer.set_data_array(pred_to_save.cpu().numpy())
                writer.set_metadata(val_data_decollated["label_meta_dict"])
                writer.write(f"{patient_id}_pred.nii.gz")

            current_mean_dice = dice_metric_mean.aggregate().item()
            current_et_dice = dice_metric_et.aggregate().item()
            current_tc_dice = dice_metric_tc.aggregate().item()
            current_wt_dice = dice_metric_wt.aggregate().item()

            dice_metric_mean.reset()
            dice_metric_et.reset()
            dice_metric_tc.reset()
            dice_metric_wt.reset()

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

    new_log_entry_df = pd.DataFrame([current_epoch_log])
    log_df = pd.concat([log_df, new_log_entry_df], ignore_index=True)
    try:
        log_df.to_csv(training_log_csv_path, index=False)
    except Exception as e:
        print(f"Error saving training log: {e}")

    if (epoch + 1) % SAVE_CHECKPOINT_FREQ == 0 or (epoch + 1) == MAX_EPOCHS:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'patience_counter': patience_counter,
            'best_val_metric_for_early_stopping': best_val_metric_for_early_stopping,
            'epoch_loss_values': epoch_loss_values,
            'metric_values_mean': metric_values_mean,
            'metric_values_et': metric_values_et,
            'metric_values_tc': metric_values_tc,
            'metric_values_wt': metric_values_wt,
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"✅ Saved periodic checkpoint for epoch {epoch + 1} to {checkpoint_path}")

    if early_stop_triggered:
        print(f"Terminating fine-tuning early at epoch {epoch + 1} due to early stopping.")
        break

# @title Step 8: Save Final Model
torch.save(model.state_dict(), os.path.join(output_directory, "final_model_swinunetr.pth"))
print(f"Fine-tuning completed. Best Mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}.")
print(f"Models saved in: {output_directory}")


# @title Step 9: Plot Loss and Metrics (3 Subplots)

import matplotlib.pyplot as plt

plt.figure("finetune_plots", (24, 6))

# --- Plot 1: Training Loss ---
plt.subplot(1, 3, 1)
plt.title("Epoch Average Loss (Finetuning)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
train_epochs = range(1, len(epoch_loss_values) + 1)
plt.plot(train_epochs, epoch_loss_values, color='red', label="Train Loss")
plt.grid(True)
plt.legend()

# --- Plot 2: Individual Validation Dice Scores (ET, TC, WT) ---
plt.subplot(1, 3, 2)
plt.title("Validation Dice Scores (ET, TC, WT) (Finetuning)")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
val_epochs = [(i + 1) * VAL_INTERVAL for i in range(len(metric_values_et))]
plt.plot(val_epochs, metric_values_et, color='red', marker='s', linestyle='--', label="ET Dice")
plt.plot(val_epochs, metric_values_tc, color='green', marker='^', linestyle='-.', label="TC Dice")
plt.plot(val_epochs, metric_values_wt, color='purple', marker='d', linestyle=':', label="WT Dice")
plt.grid(True)
plt.legend()
plt.ylim([0, 1])

# --- Plot 3: Overall Mean Dice Score ---
plt.subplot(1, 3, 3)
plt.title("Validation Mean Dice (Finetuning)")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(val_epochs, metric_values_mean, color='blue', marker='o', linestyle='-', label="Mean Dice")
plt.grid(True)
plt.legend()
plt.ylim([0, 1])

plt.suptitle("Fine-tuning Training and Validation Curves", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_directory, "finetune_training_curves_swinunetr.png"))
print("✅ Fine-tuning curves plot with 3 subplots saved.")