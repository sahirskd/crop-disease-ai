"""
CropSense AI — EfficientNetB3 Training Pipeline
Dataset: PlantVillage (54K images, 38 classes)
Target: ~98% validation accuracy
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path
import json
from tqdm import tqdm


# ─── Config ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/PlantVillage", type=str)
    parser.add_argument("--output_dir", default="experiments/", type=str)
    parser.add_argument("--model", default="efficientnet_b3", type=str,
                        choices=["efficientnet_b0", "efficientnet_b3", "resnet50"])
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--img_size", default=300, type=int)
    parser.add_argument("--val_split", default=0.15, type=float)
    parser.add_argument("--test_split", default=0.10, type=float)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--freeze_epochs", default=5, type=int,
                        help="Epochs to train only the head before unfreezing backbone")
    return parser.parse_args()


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int, phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_dataloaders(data_dir: str, img_size: int, batch_size: int,
                    val_split: float, test_split: float):
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    n = len(full_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms per split
    train_ds.dataset.transform = get_transforms(img_size, "train")
    val_ds.dataset.transform = get_transforms(img_size, "val")
    test_ds.dataset.transform = get_transforms(img_size, "test")

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=pin_memory),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=pin_memory),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=pin_memory),
    }
    return loaders, class_names


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, device: torch.device):
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def freeze_backbone(model, model_name: str):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B init
    if args.use_wandb:
        wandb.init(project="cropsense-ai", config=vars(args))

    # Data
    loaders, class_names = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size,
        args.val_split, args.test_split
    )
    num_classes = len(class_names)
    print(f"Classes: {num_classes} | Train: {len(loaders['train'].dataset)}")

    # Save class names for inference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Model
    model = build_model(args.model, num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(str(device) == "cuda"))

    # Phase 1: Train head only
    freeze_epochs = min(args.freeze_epochs, args.epochs)
    fine_tune_epochs = args.epochs - freeze_epochs

    best_val_acc = 0.0
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    if freeze_epochs > 0:
        freeze_backbone(model, args.model)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs)

        print(f"\n── Phase 1: Training head for {freeze_epochs} epochs ──")
        for epoch in range(freeze_epochs):
            train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler)
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()
            print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                           "val_loss": val_loss, "val_acc": val_acc, "phase": 1})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")

    # Phase 2: Fine-tune full model
    if fine_tune_epochs > 0:
        unfreeze_all(model)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs)

        print(f"\n── Phase 2: Full fine-tuning for {fine_tune_epochs} epochs ──")
        for epoch in range(fine_tune_epochs):
            train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler)
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()
            print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                           "val_loss": val_loss, "val_acc": val_acc, "phase": 2})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")

    # Final test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    print(f"\n🏁 Test Accuracy: {test_acc:.4f}")

    # Save metadata
    metadata = {
        "model": args.model,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if args.use_wandb:
        wandb.finish()

    print(f"\n✅ Done. Model saved to {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
