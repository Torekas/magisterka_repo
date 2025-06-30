## weighted cross-entropy plus 0.5x Dice loss

from comet_ml import Experiment  # Comet-ML for experiment logging

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

import albumentations as A                # For fast image augmentations
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------
#                Normalization
# ---------------------------------------------
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------
#            Loss Definitions
# ---------------------------------------------
class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation. Promotes overlap between prediction and ground truth."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        num_cls    = logits.shape[1]
        prob       = F.softmax(logits, dim=1)
        tgt_onehot = F.one_hot(target, num_classes=num_cls).permute(0,3,1,2).float()
        inter      = (prob * tgt_onehot).sum((2,3))
        union      = prob.sum((2,3)) + tgt_onehot.sum((2,3))
        dice       = (2*inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()  # Higher dice means better overlap, so loss is 1-dice

def get_loss(num_classes, device):
    """Returns a combined loss: weighted cross-entropy + 0.5x Dice."""
    weights = torch.tensor([1.0, 1.0, 2.0], device=device)  # Heavier penalty for class 2 (lane)
    ce      = nn.CrossEntropyLoss(weight=weights)
    dice    = DiceLoss()
    return lambda logits, masks: ce(logits, masks) + 0.5 * dice(logits, masks)

# ---------------------------------------------
#            Dataset Definition
# ---------------------------------------------
class RoadLaneDataset(Dataset):
    """Custom dataset for road (class 1) and lane (class 2) segmentation."""

    def __init__(self, image_dir, road_mask_dir, lane_mask_dir,
                 transform=None, target_size=(1392, 512)):
        self.image_dir      = image_dir
        self.road_mask_dir  = road_mask_dir
        self.lane_mask_dir  = lane_mask_dir
        self.transform      = transform
        self.target_size    = target_size

        # Find all image files in directory
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and both masks
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image    = Image.open(img_path).convert("RGB")

        base       = os.path.splitext(img_name)[0]
        road_mask  = Image.open(os.path.join(self.road_mask_dir, base + ".png")).convert("L")
        lane_mask  = Image.open(os.path.join(self.lane_mask_dir, base + ".png")).convert("L")

        # Resize to target size for network
        image     = image.resize(self.target_size, Image.BILINEAR)
        road_mask = road_mask.resize(self.target_size, Image.NEAREST)
        lane_mask = lane_mask.resize(self.target_size, Image.NEAREST)

        # Create combined mask: 0 = background, 1 = road, 2 = lane
        road_arr  = (np.array(road_mask) > 0).astype(np.uint8)
        lane_arr  = (np.array(lane_mask) > 0).astype(np.uint8)
        combined  = road_arr.copy()
        combined[lane_arr == 1] = 2  # Lane pixels overwrite road

        # Apply augmentation if requested
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=combined)
            image = augmented["image"]
            mask  = augmented["mask"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.long()
        else:
            image = torchvision.transforms.functional.to_tensor(image)
            image = torchvision.transforms.functional.normalize(image, mean=IMG_MEAN, std=IMG_STD)
            mask  = torch.from_numpy(combined).long()

        return image, mask

# ---------------------------------------------
#             Model Initialisation
# ---------------------------------------------
def get_model(num_classes: int):
    """Get a DeepLabV3+ResNet50 model pre-trained, but with head for num_classes."""
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model   = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    # Replace classifier head with new number of output channels
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    # Replace aux head if present
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)
    return model

# ---------------------------------------------
#            Training & Evaluation utils
# ---------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch and return average loss."""
    model.train()
    running = 0.0
    for i, (imgs, masks) in enumerate(dataloader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outs    = model(imgs)["out"]
        loss    = criterion(outs, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
        if i % 10 == 0:
            print(f"[TRAIN] Batch {i}/{len(dataloader)}  Loss: {loss.item():.4f}")
    return running / len(dataloader)

def evaluate(model, dataloader, device, num_classes):
    """Compute intersection-over-union (IoU) for each class."""
    model.eval()
    inter = [0] * num_classes
    uni   = [0] * num_classes
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)["out"].argmax(dim=1)
            for c in range(num_classes):
                inter[c] += ((preds == c) & (masks == c)).sum().item()
                uni[c]   += ((preds == c) | (masks == c)).sum().item()
    return [(inter[c] / uni[c] if uni[c] > 0 else 0.0) for c in range(num_classes)]

# ---------------------------------------------
#            Checkpoint helpers
# ---------------------------------------------
def save_checkpoint(state: dict, filename: str):
    """Save model/optimizer state to disk."""
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename: str, device):
    """Resume training from a checkpoint."""
    if not os.path.exists(filename):
        return 1
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    start = ckpt["epoch"] + 1
    print(f"Loaded checkpoint '{filename}' (resuming from epoch {start})")
    return start

# ---------------------------------------------
#                    Main
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepLabV3 road & lane segmentation with Albumentations + DiceLoss")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Comet-ML experiment initialization
    experiment = Experiment(
        api_key="Zfgk7BYvTUyWQWSh7g1M8SSt2",
        project_name="segmentation",
        workspace="torekas"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 3         # bg, road, lane
    batch_size  = 4
    lr          = 1e-3
    epochs      = args.epochs

    # Log experiment hyperparameters
    experiment.log_parameters({
        "epochs":      epochs,
        "batch_size":  batch_size,
        "learning_rate": lr,
        "loss":        "Weighted CE + 0.5*Dice",
        "augmentations": "RndBrightnessContrast,GaussNoise,CLAHE"
    })

    # ---------------------------------------------
    #        Albumentations pipelines
    # ---------------------------------------------
    train_transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2()
    ])

    # Paths for images and masks
    root = r"C:\Users\janmi\PycharmProjects\Magisterka\night_segmentation"
    train_image_dir = os.path.join(root, "images", "train")
    val_image_dir   = os.path.join(root, "images", "val")
    train_road_dir  = os.path.join(root, "da_seg_annotations", "train")
    val_road_dir    = os.path.join(root, "da_seg_annotations", "val")
    train_lane_dir  = os.path.join(root, "ll_seg_annotations", "train")
    val_lane_dir    = os.path.join(root, "ll_seg_annotations", "val")

    # Directory for saving model checkpoints
    ckpt_dir = os.path.join(root, "checkpoints_deeplab_alb_dice")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = os.path.join(ckpt_dir, "latest_checkpoint.pth")

    # Dataset and DataLoader construction
    train_set = RoadLaneDataset(train_image_dir, train_road_dir, train_lane_dir, transform=train_transform)
    val_set   = RoadLaneDataset(val_image_dir,   val_road_dir,   val_lane_dir,   transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    experiment.log_text(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    model     = get_model(num_classes).to(device)
    criterion = get_loss(num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    # Optionally resume from checkpoint
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, latest_ckpt, device)
        if start_epoch > epochs:
            print("Already finished training.")
            return

    for ep in range(start_epoch, epochs+1):
        print(f"\n=== Epoch {ep}/{epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        iou        = evaluate(model, val_loader, device, num_classes)
        road_iou, lane_iou = iou[1], iou[2]
        miou       = (road_iou + lane_iou) / 2.0

        print(f"Epoch {ep} → loss {train_loss:.4f} | road IoU={road_iou:.3f} lane IoU={lane_iou:.3f} mIoU={miou:.3f}")

        # Log metrics to Comet-ML
        experiment.log_metric("train_loss", train_loss, step=ep)
        experiment.log_metric("road_iou",   road_iou,   step=ep)
        experiment.log_metric("lane_iou",   lane_iou,   step=ep)
        experiment.log_metric("mean_iou",   miou,       step=ep)

        # Save model checkpoints
        state = {
            "epoch":       ep,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }
        save_checkpoint(state, os.path.join(ckpt_dir, f"epoch_{ep}.pth"))
        save_checkpoint(state, latest_ckpt)

    # Save the final model weights
    final_path = os.path.join(root, "deeplabv3_alb_dice_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Model saved to {final_path}")
    experiment.log_model("deeplabv3_alb_dice_final", final_path)

if __name__ == "__main__":
    main()
