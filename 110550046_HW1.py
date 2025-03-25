"""
Image classification training and evaluation script using PyTorch.

This module defines a dataset class, applies data augmentation, and trains
and evaluates a ResNet152 model on an image dataset with 100 classes.
"""

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
from tqdm import tqdm


class SimpleImageDataset(Dataset):
    """Custom dataset for loading images from directory structure."""

    def __init__(self, root_dir, transform=None, test_mode=False):
        """Initialize the dataset with image paths and labels."""
        self.transform = transform
        self.test_mode = test_mode
        self.image_paths = []
        self.labels = []
        self.filenames = []

        if test_mode:
            self.filenames = sorted(os.listdir(root_dir))
            self.image_paths = [os.path.join(root_dir, f) for f in self.filenames]
        else:
            for class_dir in sorted(os.listdir(root_dir)):
                label = int(class_dir)
                class_path = os.path.join(root_dir, class_dir)
                if os.path.isdir(class_path):
                    for fname in os.listdir(class_path):
                        self.image_paths.append(os.path.join(class_path, fname))
                        self.labels.append(label)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieve image and label or image and name for test mode."""
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.test_mode:
            name = os.path.splitext(self.filenames[idx])[0]
            return img, name
        return img, self.labels[idx]


def get_transform(train=True):
    """Return data augmentation or normalization transform."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(500),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def main():
    """Run model training, validation, and inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("test", exist_ok=True)

    for label in tqdm(range(100)):
        src_folder = os.path.join("val", str(label))
        for fname in os.listdir(src_folder):
            src = os.path.join(src_folder, fname)
            dst_name = f"{label}_{fname}"
            dst = os.path.join("test", dst_name)
            shutil.copy2(src, dst)

    full_dataset = SimpleImageDataset("train", transform=get_transform(True))
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42,
    )
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        SimpleImageDataset("test", get_transform(False), test_mode=True),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 100),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = total_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"
            ):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(
            f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, "
            f"Val Acc = {val_acc:.2f}%"
        )
        scheduler.step(val_acc)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    report = classification_report(all_labels, all_preds, digits=4)
    print("Classification Report:\n")
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs, names in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            predictions.extend(zip(names, preds))

    pd.DataFrame(predictions, columns=["image_name", "pred_label"]).to_csv(
        "prediction.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
