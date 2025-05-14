import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from torchvision import transforms
import random


# Информация об авторе проекта:
#  ФИО: Погорельских Константин Владимирович
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21


# Устройство (CPU или GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClagnoscoDataset(torch.utils.data.Dataset):
    '''
    Класс для загрузки изображений.
    '''
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

        image = Image.open(image_resize_path)
        image_square = Image.open(image_square_path)

        return {
            'uid': uid,
            'image': image,
            'image_width': self.data.loc[idx, 'image_width'],
            'image_height': self.data.loc[idx, 'image_height'],
            'image_square': image_square,
        }
    
    def bucketize(self):
        '''
        Разделение на бакеты по разрешению изображения.
        '''
        buckets = dict()
        for idx, row in self.data.iterrows():
            key = (row['image_width'], row['image_height'])
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)
        self.buckets = buckets
        return buckets
    
    def bucket_counting(self):
        '''
        Подсчёт количества изображений в каждом бакете.
        '''
        return sorted([(i, len(j)) for i, j in self.buckets.items()], key=lambda x: -x[1])
    
    def random_splitting_batching_buckets(self, percent=0.75, seed=42, batch_size=-1, max_batch_size=32):
        '''
        Разделение на 2 выборки (тренировочную и тестовую) с батчами. 0 и 1 на самом деле работают.
        
        - batch_size = -1 -- Размер бытча: Автоматическое гармоническое среднее (по умолчанию)
        - batch_size = 0  -- Размер бытча: Без батчей
        '''
        bucket_count = [i[1] for i in self.bucket_count]

        def harmonic_mean(array, percent, max_batch_size):
            len_array = len(array)
            if percent == 0 or percent == 1:
                batch_size = round(len_array / sum(1/x for x in array if x > 0))
            else:
                batch_size = round(len_array / sum(1/(x*percent) for x in array if x > 0))
            if batch_size == 0:
                batch_size = 1
            if batch_size > max_batch_size:
                batch_size = max_batch_size
            return batch_size

        rev_percent = 1 - percent
        if batch_size == -1:
            batch_size1 = harmonic_mean(bucket_count, percent, max_batch_size)
            batch_size2 = harmonic_mean(bucket_count, rev_percent, max_batch_size)
        else:
            if batch_size > max_batch_size:
                batch_size = max_batch_size
            batch_size1 = batch_size
            batch_size2 = batch_size
        
        part1 = []
        part2 = []
        for wh, bucket in self.buckets.items():
            random.seed(seed)
            random.shuffle(bucket)
            splitting_part = int(len(bucket) * rev_percent)

            this_split = bucket[splitting_part:]
            this_split_len = len(this_split)
            if this_split_len != 0:
                if batch_size1 != 0:
                    for i in range(0, this_split_len, batch_size1):
                        part1.append([wh, this_split[i:i+batch_size1]])
                else:
                    part1.append([wh, this_split])
            
            this_split = bucket[:splitting_part]
            this_split_len = len(this_split)
            if this_split_len != 0:
                if batch_size2 != 0:
                    for i in range(0, this_split_len, batch_size2):
                        part2.append([wh, this_split[i:i+batch_size2]])
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
    '''
    Изображения преобразованные в тензоры.
    '''
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Использование ToTensor() для преобразования в тензор
        if self.transform:
            item['image'] = self.transform(item['image'])
            item['image_square'] = self.transform(item['image_square'])
        
        return item


def iterate_batched_buckets(transformed_dataset, batched_buckets):
    """
    Входные данные:
        - transformed_dataset: экземпляр TransformedClagnoscoDataset
        - batched_buckets: список [(ширина, высота), [idx1, idx2, ...]] батчей
    
    Выходные данные (yield):
        - batch: словарь элементов батча
        - resolution: (ширина, высота)
    """
    for resolution, index_batch in batched_buckets:
        batch_items = [transformed_dataset[i] for i in index_batch]

        batch = {}
        for key in batch_items[0]:
            values = [item[key] for item in batch_items]
            if isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except Exception:
                    batch[key] = values
            else:
                batch[key] = values

        yield batch, resolution

def batch_buckets_to_list(batches):
    """
    Преобразует батчи в список индексов.
    """
    list_idxs = []
    for batch in [i[1] for i in batches]:
        list_idxs.extend(batch)
    return list_idxs
