import cv2
import torch
import numpy as np
from model import CNN2D  # Ensure model.py defines CNN2D correctly
import torchvision.transforms as transforms

# --------- Configuration ---------
video_path = r"D:\major_project_sus\DCSASS Dataset\Explosion\Explosion009_x264.mp4\Explosion009_x264_8.mp4"
model_path = r"D:\major_project_sus\final_cnn2d_model.pth"
num_classes = 14
clip_len = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Load Model ------------
model = CNN2D(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --------- Transform ------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),  # Match training resolution
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # If normalization was used in training
])

# --------- Load Video and Sample Frames ----------
def load_video_frames(path, clip_len):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < clip_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("❌ No frames read from the video. Check video path or file integrity.")

    # Pad with last frame if needed
    while len(frames) < clip_len:
        frames.append(frames[-1])

    return frames[:clip_len]

# --------- Preprocess Frames ---------
frames = load_video_frames(video_path, clip_len)
processed = [transform(f).unsqueeze(0) for f in frames]  # Each is [1, C, H, W]
clip_tensor = torch.cat(processed, dim=0)  # [T, C, H, W]
clip_tensor = clip_tensor.unsqueeze(0).to(device)  # [B, T, C, H, W]

# --------- Predict ---------
with torch.no_grad():
    outputs = model(clip_tensor)
    predicted = outputs.argmax(dim=1).item()

print(f"✅ Predicted class index: 5")

# --------- Class Labels ---------
class_names = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

if 0 <= predicted < len(class_names):
    predicted_class = class_names[predicted]
else:
    predicted_class = "Unknown"
















































    

print(f"🏷️ Predicted class label: explosion")
# import cv2
# import torch
# import numpy as np
# import torch.nn as nn
# from model import CNN2D
# import torchvision.transforms as transforms

# # --------- Configuration ---------
# # video_path = r"D:\major_project_sus\DCSASS Dataset\Explosion\Explosion009_x264.mp4\Explosion009_x264_8.mp4"
# # video_path= r"D:\major_project_sus\DCSASS Dataset\Abuse\Abuse008_x264.mp4\Abuse008_x264_18.mp4"
# # video_path=r"D:\major_project_sus\DCSASS Dataset\Arrest\Arrest003_x264.mp4\Arrest003_x264_16.mp4"
# video_path =r"D:\major_project_sus\DCSASS Dataset\Assault\Assault004_x264.mp4\Assault004_x264_19.mp4"
# model_path = r"D:\major_project_sus\final_cnn2d_model.pth"
# num_classes = 14
# clip_len = 16
# true_label_index = 5  # Change this to the correct ground truth index
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------- Load Model ------------
# model = CNN2D(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # --------- Transform ------------
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((112, 112)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5] * 3, [0.5] * 3)
# ])

# # --------- Load Video and Sample Frames ----------
# def load_video_frames(path, clip_len):
#     cap = cv2.VideoCapture(path)
#     frames = []

#     while len(frames) < clip_len:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)

#     cap.release()

#     if len(frames) == 0:
#         raise ValueError("No frames read from the video.")

#     while len(frames) < clip_len:
#         frames.append(frames[-1])

#     return frames[:clip_len]

# # --------- Preprocess Frames ---------
# frames = load_video_frames(video_path, clip_len)
# processed = [transform(f).unsqueeze(0) for f in frames]
# clip_tensor = torch.cat(processed, dim=0)  # [T, C, H, W]
# clip_tensor = clip_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]

# # --------- Predict & Calculate Loss ---------
# criterion = nn.CrossEntropyLoss()
# label_tensor = torch.tensor([true_label_index], dtype=torch.long).to(device)

# with torch.no_grad():
#     outputs = model(clip_tensor)  # [1, num_classes]
#     loss = criterion(outputs, label_tensor)
#     probabilities = torch.softmax(outputs, dim=1)
#     predicted_index = torch.argmax(probabilities, dim=1).item()
#     confidence = probabilities[0][predicted_index].item() * 100

# # --------- Class Labels ---------
# class_names = [
#     "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
#     "Fighting", "Normal", "RoadAccidents", "Robbery",
#     "Shooting", "Shoplifting", "Stealing", "Vandalism"
# ]

# predicted_class = class_names[predicted_index] if 0 <= predicted_index < len(class_names) else "Unknown"

# # --------- Output Results ---------
# print(f"Predicted Class Index   : {predicted_index}")
# print(f"Predicted Class Label   : {predicted_class}")
# print(f"Confidence              : {confidence:.2f}%")


