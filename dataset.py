import os
import torch
from torch.utils.data import Dataset
import random

class VideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.video_folders = []
        self.labels = []

        # Map folder names (classes) to numeric labels
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir)))}

        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)
            if not os.path.isdir(label_folder):
                continue
            for video_folder in os.listdir(label_folder):
                video_path = os.path.join(label_folder, video_folder)
                if os.path.isdir(video_path):
                    frame_files = [f for f in os.listdir(video_path) if f.endswith('.pt')]
                    if len(frame_files) >= clip_len:
                        self.video_folders.append(video_path)
                        self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_path = self.video_folders[idx]
        label = self.labels[idx]

        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.pt')])
        total_frames = len(frame_files)

        # Safety check — should already be handled in __init__
        if total_frames < self.clip_len:
            # fallback to next video
            new_idx = (idx + 1) % len(self.video_folders)
            return self.__getitem__(new_idx)

        start_idx = random.randint(0, total_frames - self.clip_len)
        clip = []
        for i in range(start_idx, start_idx + self.clip_len):
            frame_tensor = torch.load(os.path.join(video_path, frame_files[i]))
            clip.append(frame_tensor)

        clip = torch.stack(clip)  # [clip_len, C, H, W]
        return clip, label
