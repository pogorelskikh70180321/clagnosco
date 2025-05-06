import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
from torchvision import transforms
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClagnoscoDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base_dir = r"C:\!project-dataset"
        metadata_path = os.path.join(self.base_dir, "metadata.csv")
        self.data = pd.read_csv(metadata_path)
        self.buckets = self.bucketize()
        self.bucket_count = self.bucket_counting()

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
    
    def bucket_counting(self):
        return sorted([(i, len(j)) for i, j in self.buckets.items()], key=lambda x: -x[1])
    
    def random_splitting_batching_buckets(self, percent=0.75, seed=42, batch=-1, max_batch=32):
        '''
        batch = -1 -- Automatic harmonic mean
        batch = 0  -- No batches
        '''
        bucket_count = [i[1] for i in self.bucket_count]

        def harmonic_mean(array, percent, max_batch):
            len_array = len(array)
            if percent == 0 or percent == 1:
                batch = round(len_array / sum(1/x for x in array if x > 0))
            else:
                batch = round(len_array / sum(1/(x*percent) for x in array if x > 0))
            if batch == 0:
                batch = 1
            if max_batch < batch:
                batch = max_batch
            return batch
        
        if max_batch < batch:
            batch = max_batch

        part1 = []
        part2 = []
        rev_percent = 1 - percent
        is_harmonic = True if batch == -1 else False
        if is_harmonic:
            batch1 = harmonic_mean(bucket_count, percent, max_batch)
            batch2 = harmonic_mean(bucket_count, rev_percent, max_batch)
        else:
            batch1 = batch
            batch2 = batch
        for wh, bucket in self.buckets.items():
            random.seed(seed)
            random.shuffle(bucket)
            splitting_part = int(len(bucket) * rev_percent)

            this_split = bucket[splitting_part:]
            this_split_len = len(this_split)
            if this_split_len != 0:
                if batch1 != 0:
                    for i in range(0, this_split_len, batch1):
                        part1.append([wh, this_split[i:i+batch1]])
                else:
                    part1.append([wh, this_split])
            
            this_split = bucket[:splitting_part]
            this_split_len = len(this_split)
            if this_split_len != 0:
                if batch2 != 0:
                    for i in range(0, this_split_len, batch2):
                        part2.append([wh, this_split[i:i+batch2]])
                else:
                    part2.append([wh, this_split])
            seed += 1
        random.seed(seed + 1)
        random.shuffle(part1)
        random.seed(seed + 2)
        random.shuffle(part2)
        random.seed()
        return part1, part2


class TransformedClagnoscoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Apply transform to convert to tensor
        if self.transform:
            item['image'] = self.transform(item['image'])
            item['image_square'] = self.transform(item['image_square'])
        
        # Calculate aspect ratio for the model
        ratio = torch.tensor([[item['image_width'] / item['image_height']]], dtype=torch.float32).to(DEVICE)
        
        return item, ratio


if __name__ == "__main__":
    ds = ClagnoscoDataset()
    tds = TransformedClagnoscoDataset(ds)
    # bucket_count = ds.bucket_count
    # item['image']
    print(tds[0])