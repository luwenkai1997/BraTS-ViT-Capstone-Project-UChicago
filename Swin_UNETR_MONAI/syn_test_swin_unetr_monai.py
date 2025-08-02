import os
import json
import multiprocessing as mp # Import early
import torch
import numpy as np
import monai
from monai.config import print_config
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ConcatItemsd,
    EnsureTyped,
    CropForegroundd
)
from monai.data.image_writer import NibabelWriter
from monai.utils import set_determinism

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION (Modify these paths and parameters) ---

# 1. Paths
TRAINING_OUTPUT_DIR = '/home/cc/swin_unetr_train/SwinUNETR_Output_Patch96_Syn' # <--- MODIFY if different
TEST_FILES_LIST_PATH = os.path.join(TRAINING_OUTPUT_DIR, "test_set_patient_files.json")
BEST_MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, "best_metric_model_swinunetr.pth")
TEST_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_DIR, "test_run_results")
PREDICTIONS_SAVE_DIR = os.path.join(TEST_RESULTS_DIR, "predicted_segmentations")

# 2. Model & Data Parameters (Should match your training setup)
IMG_SIZE = (96, 96, 96)
INPUT_CHANNELS = 4
OUTPUT_CLASSES = 4
TARGET_SPACING = (1.0, 1.0, 1.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODALITY_KEYS = ["t1c", "t1n", "t2f", "t2w"]
LABEL_KEY = "label"
label_file_suffix = "-seg.nii.gz" # Used for patient_id extraction

# 3. Inference Parameters
SW_BATCH_SIZE = 2
OVERLAP = 0.5
# --- END CONFIGURATION ---

def main():
    set_determinism(seed=0)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_SAVE_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Loading test file list from: {TEST_FILES_LIST_PATH}")

    # --- Load Test File List ---
    with open(TEST_FILES_LIST_PATH, 'r') as f:
        test_files_to_use = json.load(f)

    # --- Define Test Transforms and DataLoader ---
    all_keys_to_load_test = MODALITY_KEYS + [LABEL_KEY]
    test_transforms = Compose(
        [
            LoadImaged(keys=all_keys_to_load_test, image_only=False, ensure_channel_first=True),
            EnsureTyped(keys=all_keys_to_load_test, dtype=torch.float32, track_meta=True),
            Orientationd(keys=all_keys_to_load_test, axcodes="RAS"),
            Spacingd(
                keys=all_keys_to_load_test,
                pixdim=TARGET_SPACING,
                mode=["bilinear"] * len(MODALITY_KEYS) + ["nearest"],
            ),
            ScaleIntensityRanged(keys=["t1c"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["t1n"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["t2f"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["t2w"], a_min=0, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )
    test_ds = Dataset(data=test_files_to_use, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    # --- Load Model ---
    model = SwinUNETR(in_channels=INPUT_CHANNELS, out_channels=OUTPUT_CLASSES, feature_size=48, use_checkpoint=False).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Metrics and Post-processing ---
    dice_metric_mean = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_et = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_tc = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_wt = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    post_pred_metric = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=OUTPUT_CLASSES)])
    post_label_metric = Compose([AsDiscrete(to_onehot=OUTPUT_CLASSES)])
    post_pred_save = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    # --- Inference Loop ---
    per_sample_results = []
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            test_input_image = batch_data["image"].to(DEVICE)
            test_ground_truth_label = batch_data[LABEL_KEY].to(DEVICE)

            patient_id = f"test_sample_{i:03d}"
            try:
                filename_data = batch_data.get(LABEL_KEY + "_meta_dict")["filename_or_obj"]
                actual_filepath_str = filename_data[0] if isinstance(filename_data, list) else filename_data
                patient_id = os.path.basename(actual_filepath_str).split(label_file_suffix)[0]
            except Exception as e:
                print(f"Warning: Could not get patient ID for sample {i}. Using fallback. Error: {e}")

            print(f"Inferring on test sample: {patient_id} ({i+1}/{len(test_loader)})...")

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                prediction = sliding_window_inference(
                    inputs=test_input_image, roi_size=IMG_SIZE, sw_batch_size=SW_BATCH_SIZE,
                    predictor=model, overlap=OVERLAP, mode="gaussian"
                )

            pred_one_hot = [post_pred_metric(p) for p in decollate_batch(prediction)]
            label_one_hot = [post_label_metric(l) for l in decollate_batch(test_ground_truth_label)]

            # Extract the single tensor from the list (batch size is 1)
            output_tensor = pred_one_hot[0]
            label_tensor = label_one_hot[0]

            # --- Calculate ET, TC, WT from one-hot masks ---
            # NOTE: Assumes model output channels map to: 1=NCR/NET, 2=ED, 3=ET
            # Enhancing Tumor (ET) is channel 3
            et_pred_mask = output_tensor[3:4].bool()
            et_label_mask = label_tensor[3:4].bool()

            # Tumor Core (TC) is channels 1 (NCR/NET) + 3 (ET)
            tc_pred_mask = output_tensor[1:2].bool() | output_tensor[3:4].bool()
            tc_label_mask = label_tensor[1:2].bool() | label_tensor[3:4].bool()

            # Whole Tumor (WT) is channels 1 (NCR/NET) + 2 (ED) + 3 (ET)
            wt_pred_mask = output_tensor[1:2].bool() | output_tensor[2:3].bool() | output_tensor[3:4].bool()
            wt_label_mask = label_tensor[1:2].bool() | label_tensor[2:3].bool() | label_tensor[3:4].bool()

            # --- Calculate all Dice scores for this sample ---
            dice_metric_mean(y_pred=pred_one_hot, y=label_one_hot)
            dice_mean = dice_metric_mean.aggregate().item()
            dice_metric_mean.reset()

            dice_metric_et(y_pred=[et_pred_mask], y=[et_label_mask])
            dice_et = dice_metric_et.aggregate().item()
            dice_metric_et.reset()

            dice_metric_tc(y_pred=[tc_pred_mask], y=[tc_label_mask])
            dice_tc = dice_metric_tc.aggregate().item()
            dice_metric_tc.reset()

            dice_metric_wt(y_pred=[wt_pred_mask], y=[wt_label_mask])
            dice_wt = dice_metric_wt.aggregate().item()
            dice_metric_wt.reset()

            # --- Helper function to convert Dice to IoU ---
            def dice_to_iou(dice_score):
                return dice_score / (2 - dice_score + 1e-8)

            # Append all results to our list
            per_sample_results.append({
                "patient_id": patient_id,
                "dice_mean": dice_mean,
                "dice_et": dice_et,
                "dice_tc": dice_tc,
                "dice_wt": dice_wt,
                "iou_mean": dice_to_iou(dice_mean),
                "iou_et": dice_to_iou(dice_et),
                "iou_tc": dice_to_iou(dice_tc),
                "iou_wt": dice_to_iou(dice_wt),
            })
            print(f"  Sample: {patient_id}, Mean Dice: {dice_mean:.4f}, ET: {dice_et:.4f}, TC: {dice_tc:.4f}, WT: {dice_wt:.4f}")

            # Code for saving the .nii.gz file would go here if needed

    print("\n✅ Inference loop completed.")

    # --- Convert results to DataFrame and save to CSV ---
    results_df = pd.DataFrame(per_sample_results)
    csv_save_path = os.path.join(TEST_RESULTS_DIR, "swin_unetr_metrics_summary_detailed.csv")
    results_df.to_csv(csv_save_path, index=False)
    print(f"✅ Detailed metrics for {len(results_df)} samples saved to: {csv_save_path}")

    # --- Generate plots from the saved CSV file ---
    print("\n--- Generating Distribution Plots ---")
    sns.set(style="whitegrid")

    # Define metrics to plot
    metrics_to_plot = {
        "dice_mean": "Mean Dice Score",
        "dice_et": "ET Dice Score",
        "dice_tc": "TC Dice Score",
        "dice_wt": "WT Dice Score"
    }

    # Create a 2x2 grid for Dice plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel() # Flatten the 2x2 grid to a 1D array

    for i, (metric_key, title) in enumerate(metrics_to_plot.items()):
        sns.histplot(results_df[metric_key], kde=True, bins=30, color="skyblue", edgecolor="black", ax=axes[i])
        axes[i].set_title(f"{title} Distribution (Swin UNETR)")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")
        mean_val = results_df[metric_key].mean()
        axes[i].axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.3f}")
        axes[i].legend()

    plt.tight_layout()
    dice_plot_path = os.path.join(TEST_RESULTS_DIR, "dice_score_distributions.png")
    plt.savefig(dice_plot_path)
    print(f"Dice distribution plots saved to: {dice_plot_path}")
    plt.show()

    # Create a similar 2x2 grid for IoU plots
    # (This part is repetitive, you could make a function for it)
    fig_iou, axes_iou = plt.subplots(2, 2, figsize=(16, 10))
    axes_iou = axes_iou.ravel()

    for i, (metric_key, title) in enumerate(metrics_to_plot.items()):
        iou_key = metric_key.replace("dice", "iou")
        iou_title = title.replace("Dice", "IoU")
        sns.histplot(results_df[iou_key], kde=True, bins=30, color="lightgreen", edgecolor="black", ax=axes_iou[i])
        axes_iou[i].set_title(f"{iou_title} Distribution (Swin UNETR)")
        axes_iou[i].set_xlabel("Score")
        axes_iou[i].set_ylabel("Frequency")
        mean_val = results_df[iou_key].mean()
        axes_iou[i].axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.3f}")
        axes_iou[i].legend()

    plt.tight_layout()
    iou_plot_path = os.path.join(TEST_RESULTS_DIR, "iou_score_distributions.png")
    plt.savefig(iou_plot_path)
    print(f"IoU distribution plots saved to: {iou_plot_path}")
    plt.show()


if __name__ == '__main__':
    try:
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method is None:
            mp.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method set to 'spawn'.")
        elif current_start_method != 'spawn':
            print(f"⚠️ Warning: Multiprocessing start method was already '{current_start_method}'. Attempting to force to 'spawn'.")
            mp.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method re-set to 'spawn'.")
        else:
            print("✅ Multiprocessing start method already 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" in str(e):
            print(f"⚠️ Multiprocessing context already set (was '{mp.get_start_method()}'). This is usually fine if set correctly earlier.")
        else:
            print(f"🚨 Error setting multiprocessing start method: {e}. Trying to continue...")
    except Exception as e:
        print(f"🚨 Unexpected error during multiprocessing setup: {e}. Trying to continue...")

    main()
