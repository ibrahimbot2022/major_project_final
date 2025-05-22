import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import VideoDataset
from model import CNN2D

# Hyperparameters
num_epochs = 15
batch_size = 4
learning_rate = 0.001
num_classes = 14
clip_len = 16

# Paths
train_path = "D:/major_project_sus/split_dataset/train"
val_path = "D:/major_project_sus/split_dataset/val"
test_path = "D:/major_project_sus/split_dataset/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
train_dataset = VideoDataset(train_path, clip_len=clip_len)
val_dataset = VideoDataset(val_path, clip_len=clip_len)
test_dataset = VideoDataset(test_path, clip_len=clip_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = CNN2D(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# For plotting
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_accuracy:.2f}%")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "final_cnn2d_model.pth")
        print("✅ Saved Best Model!")

print("🏁 Training Finished")

# Plotting Graphs
plt.figure(figsize=(10, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy", color="green", marker='o')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# Testing
print("\n🧪 Testing on Test Set...")
model.load_state_dict(torch.load("final_cnn2d_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for clips, labels in test_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"🧪 Test Accuracy: {test_accuracy:.2f}%")
