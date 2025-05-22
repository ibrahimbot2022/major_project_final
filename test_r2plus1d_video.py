# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from model_r2plus1d import R2Plus1D  # Ensure model file is correct

# # -------- Configuration --------
# # clip_path = r"D:\major_project_sus\preprocessed_dataset\Assault\Assault002_x264_3\10.pt"
# clip_path=r"D:\major_project_sus\preprocessed_dataset\Fighting\Fighting002_x264_12\11.pt"
# model_path = r"D:\major_project_sus\best_r2plus1d_model.pth"
# num_classes = 14
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------- Load Model --------
# model = R2Plus1D(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # -------- Load Input Clip --------
# clip = torch.load(clip_path)

# # TEMP FIX: If clip is only one frame [3, H, W], repeat it T=16 times
# if clip.ndim == 3 and clip.shape[0] == 3:
#     print("⚠️ Single frame detected. Repeating it 16 times to simulate a clip...")
#     resize = transforms.Resize((112, 112))
#     clip = resize(clip)  # Resize single frame to (3, 112, 112)
#     clip = clip.unsqueeze(0).repeat(16, 1, 1, 1)  # Shape becomes [16, 3, 112, 112]

# # Check shape: should be [T, 3, H, W]
# if clip.ndim != 4 or clip.shape[1] != 3:
#     raise ValueError(f"❌ Invalid tensor shape. Expected [T, 3, H, W], got: {clip.shape}")

# # Add batch dimension: [B, T, C, H, W]
# clip = clip.unsqueeze(0).to(device)

# # -------- Predict --------
# with torch.no_grad():
#     outputs = model(clip)
#     predicted = outputs.argmax(dim=1).item()

# # -------- Class Names --------
# class_names = [
#     "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
#     "Fighting", "Normal", "RoadAccidents", "Robbery",
#     "Shooting", "Shoplifting", "Stealing", "Vandalism"
# ]

# # -------- Result --------
# predicted_class = class_names[predicted] if predicted < len(class_names) else "Unknown"

# print(f"\n✅ Predicted class index: {predicted}")
# print(f"🏷️ Predicted class label: {predicted_class}")


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model_r2plus1d import R2Plus1D

# -------- Configuration --------
# clip_path = r"D:\major_project_sus\preprocessed_dataset\Fighting\Fighting002_x264_12\11.pt"
# clip_path =r"D:\major_project_sus\preprocessed_dataset\Explosion\Explosion004_x264_9\5.pt"
# clip_path =r"D:\major_project_sus\preprocessed_dataset\Abuse\Abuse048_x264_8\13.pt"
clip_path =r"D:\major_project_sus\preprocessed_dataset\Arrest\Arrest003_x264_16\13.pt"
model_path = r"D:\major_project_sus\best_r2plus1d_model.pth"
num_classes = 14
true_label_index = 6  # Update this to match the true label (e.g., 6 for 'Fighting')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model --------
model = R2Plus1D(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------- Load Input Clip --------
clip = torch.load(clip_path)

# Handle single frame input
if clip.ndim == 3 and clip.shape[0] == 3:
    print("⚠️ Single frame detected. Repeating it 16 times to simulate a clip...")
    resize = transforms.Resize((112, 112))
    clip = resize(clip)
    clip = clip.unsqueeze(0).repeat(16, 1, 1, 1)

# Ensure shape [T, 3, H, W]
if clip.ndim != 4 or clip.shape[1] != 3:
    raise ValueError(f"Invalid shape: Expected [T, 3, H, W], got: {clip.shape}")

# Add batch dimension: [1, T, 3, H, W]
clip = clip.unsqueeze(0).to(device)

# -------- Predict & Compute Loss --------
criterion = nn.CrossEntropyLoss()
label_tensor = torch.tensor([true_label_index], dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(clip)  # [1, num_classes]
    loss = criterion(outputs, label_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_index].item() * 100

# -------- Class Names --------
class_names = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "Unknown"

# -------- Result --------
print(f"\n✅ Predicted class index : {predicted_index}")
print(f"🏷️ Predicted class label : {predicted_class}")
print(f"📊 Confidence            : {confidence:.2f}%")
print(f"💥 Loss                  : {loss.item():.4f}")
