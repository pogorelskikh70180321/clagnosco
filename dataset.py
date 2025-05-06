import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClagnoscoDataset(Dataset):
    def __init__(self):
        self.base_dir = r"C:\!project-dataset"
        metadata_path = os.path.join(self.base_dir, "metadata.csv")
        self.data = pd.read_csv(metadata_path)
        self.buckets = self.bucketize()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid = self.data.loc[idx, 'uid']
        image_resize_path = os.path.join(self.base_dir, 'images_resize', f"{uid}.jpg")
        image_square_path = os.path.join(self.base_dir, 'images_square', f"{uid}.jpg")
        caption_path = os.path.join(self.base_dir, 'captions', f"{uid}.txt")
        embedding_path = os.path.join(self.base_dir, 'captions_emb', f"{uid}.npy")

        image = Image.open(image_resize_path).convert('RGB')
        image_square = Image.open(image_square_path).convert('RGB')
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        embedding = np.load(embedding_path)

        return {
            'uid': uid,
            'image': image,
            'image_width': self.data.loc[idx, 'image_width'],
            'image_height': self.data.loc[idx, 'image_height'],
            'image_square': image_square,
            'caption': caption,
            'embedding': torch.tensor(embedding, dtype=torch.float16).to(DEVICE)
        }
    
    def bucketize(self):
        buckets = dict()
        for idx, row in self.data.iterrows():
            key = (row['image_width'], row['image_height'])
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)
        self.buckets = buckets
        return buckets
    
    def bucket_count(self):
        return sorted([(i, len(j)) for i, j in self.buckets.items()], key=lambda x: -x[1])


if __name__ == "__main__":
    ds = ClagnoscoDataset()
    bucket_count = ds.bucket_count()
    print(bucket_count)