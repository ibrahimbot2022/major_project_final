import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import VideoDataset
from model_r2plus1d import R2Plus1D

# Hyperparameters
num_epochs = 20
batch_size = 4
learning_rate = 0.001
num_classes = 14
clip_len = 16

# Paths
train_path = "D:/major_project_sus/split_dataset/train"
val_path = "D:/major_project_sus/split_dataset/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets and loaders
train_dataset = VideoDataset(train_path, clip_len=clip_len)
val_dataset = VideoDataset(val_path, clip_len=clip_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = R2Plus1D(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Tracking for plots
train_losses, val_losses = [], []
val_accuracies = []
best_val_loss = float('inf')

# Training
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0

    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_r2plus1d_model.pth")
        print("✅ Saved Best R(2+1)D Model!")

print("🏁 Training Finished!")

# Plot graphs
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('R(2+1)D: Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('R(2+1)D: Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('r2plus1d_training_graphs.png')
plt.show()
