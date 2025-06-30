from comet_ml import Experiment

import os
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp     # For pretrained UNet and other models
import albumentations as A                    # Fast, flexible augmentations
from albumentations.pytorch import ToTensorV2

#######################################
#     Night Enhancement Function      #
#######################################
def enhance_night_image(pil_img):
    """
    Applies several image enhancement techniques for night-time images:
    - Denoising, CLAHE on Y-channel, and gamma correction.
    Returns: enhanced PIL.Image
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    # Gamma correction for brightening
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = ((np.linspace(0, 255, 256) / 255.0) ** inv_gamma) * 255
    table = table.astype(np.uint8)
    img = cv2.LUT(img, table)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#######################################
#           Dataset Class             #
#######################################
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

class RoadLaneDataset(Dataset):
    """
    Custom Dataset for (optionally enhanced) night images + road/lane masks.
    Combines two mask files into a single target: 0=background, 1=road, 2=lane.
    """
    def __init__(self, image_dir, road_mask_dir, lane_mask_dir,
                 night_enhance=True, transform=None, target_size=(1392, 512)):
        self.image_dir     = image_dir
        self.road_mask_dir = road_mask_dir
        self.lane_mask_dir = lane_mask_dir
        self.night_enhance = night_enhance
        self.transform     = transform
        self.target_size   = target_size

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and optionally enhance the image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.night_enhance:
            img = enhance_night_image(img)

        base = os.path.splitext(img_name)[0]
        road = Image.open(os.path.join(self.road_mask_dir, base + ".png")).convert('L')
        lane = Image.open(os.path.join(self.lane_mask_dir, base + ".png")).convert('L')

        # Resize everything to network input shape
        img  = img.resize(self.target_size, Image.BILINEAR)
        road = road.resize(self.target_size, Image.NEAREST)
        lane = lane.resize(self.target_size, Image.NEAREST)

        # Build combined segmentation mask
        road_arr = np.array(road) > 0
        lane_arr = np.array(lane) > 0
        mask_np  = np.zeros(self.target_size[::-1], dtype=np.int64)
        mask_np[road_arr] = 1
        mask_np[lane_arr] = 2

        # Albumentations transform or basic normalization
        if self.transform:
            augmented = self.transform(image=np.array(img), mask=mask_np)
            image = augmented['image']
            m_aug  = augmented['mask']
            mask   = torch.from_numpy(m_aug).long() if isinstance(m_aug, np.ndarray) else m_aug.long()
        else:
            image = transforms.functional.to_tensor(img)
            image = transforms.functional.normalize(image, mean=IMG_MEAN, std=IMG_STD)
            mask  = torch.from_numpy(mask_np).long()

        return image, mask

#######################################
#   Combined Loss: CE + Dice (lane↑)  #
#######################################
class DiceLoss(nn.Module):
    """
    Dice loss for multiclass segmentation (differentiable IoU proxy).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        num_cls   = logits.shape[1]
        prob      = F.softmax(logits, dim=1)
        tgt_onehot= F.one_hot(target, num_classes=num_cls).permute(0,3,1,2).float()
        inter     = (prob * tgt_onehot).sum((2,3))
        union     = prob.sum((2,3)) + tgt_onehot.sum((2,3))
        dice      = (2*inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

def get_loss(num_classes, device):
    """
    Returns a combined loss: weighted CrossEntropy + 0.5 * Dice.
    Weights penalize missing lane pixels more.
    """
    weights = torch.tensor([1.0, 1.0, 2.0], device=device)
    ce      = nn.CrossEntropyLoss(weight=weights)
    dice    = DiceLoss()
    return lambda logits, masks: ce(logits, masks) + 0.5 * dice(logits, masks)

#######################################
#      Model: Pretrained UNet34       #
#######################################
def get_unet_model(num_classes):
    """
    Returns a UNet with ResNet34 encoder, pretrained on ImageNet, and adapted for num_classes.
    """
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = num_classes,
        activation      = None
    )

#######################################
#        Checkpoint Utilities         #
#######################################
def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """
    Saves the model and optimizer state dicts to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, optimizer, checkpoint_dir, filename='checkpoint.pth'):
    """
    Loads model and optimizer state dicts from disk, resuming at the next epoch.
    Returns: epoch to start at (int)
    """
    path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=lambda s, l: s)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        return ckpt['epoch'] + 1
    else:
        return 1

#######################################
#         Training & Evaluation       #
#######################################
def train_one_epoch(model, loader, opt, loss_fn, device):
    """
    Trains for one epoch and returns average loss.
    """
    model.train()
    total_loss   = 0.0
    num_batches  = len(loader)
    for i, (imgs, masks) in enumerate(loader, 1):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, masks)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        print(f"  Iter {i}/{num_batches} - Loss: {loss.item():.4f}")
    avg_loss = total_loss / num_batches
    print(f" => Avg epoch loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_iou(model, loader, device, num_classes):
    """
    Evaluates per-class Intersection-over-Union (IoU) on validation data.
    """
    model.eval()
    ints, uns = [0]*num_classes, [0]*num_classes
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs).argmax(1)
            for c in range(num_classes):
                ints[c] += ((preds==c)&(masks==c)).sum().item()
                uns[c] += ((preds==c)|(masks==c)).sum().item()
    return [(ints[c]/uns[c] if uns[c]>0 else 0.0) for c in range(num_classes)]

#######################################
#               Main                  #
#######################################
def main():
    # --- Comet Experiment Setup ---
    experiment = Experiment(
        api_key="Zfgk7BYvTUyWQWSh7g1M8SSt2",
        project_name="segmentation",
        workspace="torekas"
    )

    # --- Device & Hyperparameters ---
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes  = 3         # background, road, lane
    epochs       = 10
    bs, lr       = 4, 1e-3

    # --- Log Hyperparameters ---
    experiment.log_parameters({
        "model":          "UNet34",
        "epochs":         epochs,
        "batch_size":     bs,
        "learning_rate":  lr,
        "num_classes":    num_classes,
        "night_enhance":  True,
        "augmentations":  "RandomBrightnessContrast,GaussNoise,CLAHE"
    })

    # --- Define Augmentations / Transforms ---
    train_tf = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2()
    ])
    val_tf   = A.Compose([
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2()
    ])

    # --- Dataset Directories ---
    root               = r"C:\Users\janmi\PycharmProjects\Magisterka\night_segmentation"
    train_image_dir    = os.path.join(root, "images", "train")
    val_image_dir      = os.path.join(root, "images", "val")
    train_road_mask_dir= os.path.join(root, "da_seg_annotations", "train")
    val_road_mask_dir  = os.path.join(root, "da_seg_annotations", "val")
    train_lane_mask_dir= os.path.join(root, "ll_seg_annotations", "train")
    val_lane_mask_dir  = os.path.join(root, "ll_seg_annotations", "val")

    # --- Create Datasets & Loaders ---
    train_ds = RoadLaneDataset(
        train_image_dir,
        train_road_mask_dir,
        train_lane_mask_dir,
        night_enhance=True,
        transform=train_tf
    )
    val_ds   = RoadLaneDataset(
        val_image_dir,
        val_road_mask_dir,
        val_lane_mask_dir,
        night_enhance=True,
        transform=val_tf
    )
    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  pin_memory=True, drop_last=True)
    vl_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False, pin_memory=True)

    # --- Log Dataset Sizes ---
    experiment.log_text(f"Train dataset size: {len(train_ds)}")
    experiment.log_text(f"Val   dataset size: {len(val_ds)}")

    # --- Model, Optimizer, Loss ---
    model   = get_unet_model(num_classes).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss(num_classes, device)

    # --- Resume from Checkpoint ---
    ckpt_dir    = "./checkpoints_unet34"
    start_epoch = load_checkpoint(model, opt, ckpt_dir)

    # --- Training Loop ---
    for e in range(start_epoch, epochs+1):
        print(f"\nEpoch {e}/{epochs}")
        tr_loss  = train_one_epoch(model, tr_loader, opt, loss_fn, device)
        ious     = evaluate_iou(model, vl_loader, device, num_classes)
        mean_iou = (ious[1] + ious[2]) / 2.0

        print(
            f"Epoch {e} Summary: "
            f"Avg Loss={tr_loss:.4f}, "
            f"Road IoU={ious[1]:.3f}, "
            f"Lane IoU={ious[2]:.3f}, "
            f"Mean IoU={mean_iou:.3f}\n"
        )

        # --- Log Metrics to Comet ---
        experiment.log_metric("train_loss", tr_loss,     step=e)
        experiment.log_metric("road_iou",   ious[1],     step=e)
        experiment.log_metric("lane_iou",   ious[2],     step=e)
        experiment.log_metric("mean_iou",   mean_iou,    step=e)

        # --- Save Checkpoint ---
        save_checkpoint({
            'epoch':       e,
            'model_state': model.state_dict(),
            'opt_state':   opt.state_dict()
        }, checkpoint_dir=ckpt_dir)

    # --- Final Model Save & Log ---
    final_path = "unet34_night_final_resized.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    experiment.log_model("unet34_night_resized", final_path)

if __name__ == "__main__":
    main()
