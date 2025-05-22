import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = os.listdir(source_dir)

    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        if not os.path.isdir(class_path):
            continue

        videos = os.listdir(class_path)
        random.shuffle(videos)

        train_end = int(train_ratio * len(videos))
        val_end = train_end + int(val_ratio * len(videos))

        splits = {
            "train": videos[:train_end],
            "val": videos[train_end:val_end],
            "test": videos[val_end:]
        }

        for split, video_list in splits.items():
            split_class_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_class_dir, exist_ok=True)

            for vid in video_list:
                src = os.path.join(class_path, vid)
                dst = os.path.join(split_class_dir, vid)
                shutil.copytree(src, dst)

        print(f"✅ Done splitting for class: {cls}")

# Run the function
source_dir = "D:/major_project_sus/preprocessed_dataset"
dest_dir = "D:/major_project_sus/split_dataset"

split_dataset(source_dir, dest_dir)
print("🎉 .pt dataset successfully split into train, val, and test!")
