import os
import csv
import torch
import random
from tqdm import tqdm
from model.UNet import UNet
from model.UNet3Plus import UNet3Plus
import matplotlib.pyplot as plt
from loss import DiceLossWithLogits
from dataset.isic import ISICDataset
from torch.utils.data import DataLoader, random_split

SEED = 42
IMAGE_SIZE = 256 #should be in {256, 320, 384, 512, 640} for UNet3Plus
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 0
VAL_RATIO = 0.2
USE_AMP = True
SAVE_DIR = ""    #Path to save checkpoints and logs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ""      #Path to dataset

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_csv(history, csv_path):
    fieldnames = [
        "epoch",
        "train_loss", "train_dice", "train_iou",
        "val_loss", "val_dice", "val_iou"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in history:
            writer.writerow(row)

def save_plots(history, save_dir):
    epochs = [x["epoch"] for x in history]

    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]

    train_dice = [x["train_dice"] for x in history]
    val_dice = [x["val_dice"] for x in history]

    train_iou = [x["train_iou"] for x in history]
    val_iou = [x["val_iou"] for x in history]

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # Dice
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training and Validation Dice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_curve.png"), dpi=200)
    plt.close()

    # IoU
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_iou, label="Train IoU")
    plt.plot(epochs, val_iou, label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iou_curve.png"), dpi=200)
    plt.close()

@torch.no_grad()
def binary_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    dice = (2 * intersection + eps) / (preds.sum() + targets.sum() + eps)
    iou = (intersection + eps) / (union + eps)

    return dice.item(), iou.item()

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        dice, iou = binary_metrics_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device == "cuda" and USE_AMP)):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice, iou = binary_metrics_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    full_train_dataset = ISICDataset(
        data_dir=DATA_DIR,
        split="train",
        image_size=IMAGE_SIZE
    )

    test_dataset = ISICDataset(
        data_dir=DATA_DIR,
        split="test",
        image_size=IMAGE_SIZE
    )

    val_size = int(len(full_train_dataset) * VAL_RATIO)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        bn=True
    ).to(DEVICE)

    # model = UNet3Plus(
    #     in_channels=3,
    #     out_channels=1
    # ).to(DEVICE)

    criterion = DiceLossWithLogits()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    # scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda" and USE_AMP))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and USE_AMP))

    best_val_dice = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )

        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, DEVICE, desc="Val"
        )

        scheduler.step(val_dice)

        print(f"Train | loss: {train_loss:.4f} | dice: {train_dice:.4f} | iou: {train_iou:.4f}")
        print(f"Val   | loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_iou:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        })

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"Saved best model: val_dice = {best_val_dice:.4f}")

    print("\nTesting best model...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE))

    test_loss, test_dice, test_iou = evaluate(
        model, test_loader, criterion, DEVICE, desc="Test"
    )

    print(f"Test  | loss: {test_loss:.4f} | dice: {test_dice:.4f} | iou: {test_iou:.4f}")

    csv_path = os.path.join(SAVE_DIR, "training.csv")
    save_csv(history, csv_path)
    save_plots(history, SAVE_DIR)

if __name__ == "__main__":
    main()