from comet_ml import Experiment  # Import Comet‑ML for experiment tracking and logging
import torch
import os
import argparse

from datasets import get_images, get_dataset, get_data_loaders  # Custom dataset utilities
from model import segformer_model                               # SegFormer model builder
from config import ALL_CLASSES, LABEL_COLORS_LIST               # Class and color config
from transformers import SegformerFeatureExtractor              # HuggingFace SegFormer
from engine import train, validate                              # Training/validation logic
from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR                # Learning rate scheduler

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Command-line arguments for hyperparameters and runtime settings
parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=10,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz',
    default=[512, 416],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
    help='enable learning rate scheduler'
)
parser.add_argument(
    '--scheduler-epochs',
    dest='scheduler_epochs',
    default=[30],
    nargs='+',
    type=int,
    help='scheduler milestones'
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # Initialize Comet‑ML experiment for logging metrics and results
    experiment = Experiment(
        api_key="Zfgk7BYvTUyWQWSh7g1M8SSt2",   # Your Comet API key
        project_name="segmentation",           # Project name in Comet dashboard
        workspace="torekas"                    # Workspace name in Comet
    )
    # Log all hyperparameters to Comet‑ML for experiment traceability
    experiment.log_parameters(vars(args))

    # Create output directories for results, model checkpoints, and predictions
    out_dir = os.path.join('outputs')
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    # Device selection: Use CUDA if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Build the SegFormer model with correct number of output classes
    model = segformer_model(classes=ALL_CLASSES).to(device)
    print(model)
    # Count and display total/learnable parameters for model size estimation
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer: AdamW recommended for transformer architectures
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Load lists of training and validation images and masks
    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path='input'
    )

    # HuggingFace feature extractor for SegFormer preprocessing
    feature_extractor = SegformerFeatureExtractor(size=(args.imgsz[1], args.imgsz[0]))

    # Build PyTorch Datasets for training and validation
    train_dataset, valid_dataset = get_dataset(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=feature_extractor
    )

    # DataLoader setup for efficient batching/shuffling
    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset,
        valid_dataset,
        args.batch
    )

    # Best model checkpointing utilities for loss and mIOU
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    # Optional multi-step learning rate scheduler (reduces LR at milestones)
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True
    )

    # Lists to track loss, pixel accuracy, and mIOU across epochs
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []

    # Training and validation loop
    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch + 1}")
        # One epoch of training
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            ALL_CLASSES
        )
        # Validation on held-out data
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            ALL_CLASSES,
            LABEL_COLORS_LIST,
            epoch,
            save_dir=out_dir_valid_preds
        )
        # Store metrics for plots
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        # Save best model (lowest loss) and best model (highest mIOU)
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        # Print metrics for quick inspection
        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},",
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f},",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )

        # Log all metrics for this epoch to Comet‑ML dashboard
        experiment.log_metric("train_epoch_loss", train_epoch_loss, step=epoch)
        experiment.log_metric("train_epoch_pixacc", train_epoch_pixacc, step=epoch)
        experiment.log_metric("train_epoch_miou", train_epoch_miou, step=epoch)
        experiment.log_metric("valid_epoch_loss", valid_epoch_loss, step=epoch)
        experiment.log_metric("valid_epoch_pixacc", valid_epoch_pixacc, step=epoch)
        experiment.log_metric("valid_epoch_miou", valid_epoch_miou, step=epoch)

        # Step the LR scheduler if enabled
        if args.scheduler:
            scheduler.step()
        print('-' * 50)

    # Save loss, pixel accuracy, and mIOU plots for later analysis
    save_plots(
        train_pix_acc, valid_pix_acc,
        train_loss, valid_loss,
        train_miou, valid_miou,
        out_dir
    )
    # Save final model weights after training
    save_model(model, out_dir, name='final_model')

    # Optionally, log the final model as an artifact to Comet‑ML for reproducibility
    final_model_path = os.path.join(out_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    experiment.log_model("final_model", final_model_path)

    print('TRAINING COMPLETE')
