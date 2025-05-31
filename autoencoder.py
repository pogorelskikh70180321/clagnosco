from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import requests
import tempfile
# from torchmetrics.image import StructuralSimilarityIndexMeasure
if __name__ == "__main__":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm


# Информация об авторе проекта:
#  ФИО: Погорельских Константин Владимирович
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21


# Папка для сохранения моделей
SAVE_FOLDER = "models/"


class ClagnoscoEncoder(nn.Module):
    '''
    Вход: Тензорные изображения в батче (B, 3, H, W)

    Выход: Латентные векторы в батче (B, 512) # 512*16
    '''
    def __init__(
        self,
        latent_dim=512,
        backbone_channels=512,
        negative_slope=0.01
    ):
        super().__init__()
        self.latent_dim_size = latent_dim 

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=backbone_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Output: (B, backbone_channels, 4, 4)

        self.lin = nn.Sequential(
            nn.Linear(backbone_channels * 4 * 4, backbone_channels * 4),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(backbone_channels * 4, latent_dim),
        )

    def forward(self, x):
        B = x.size(0)
        conved = self.conv(x)
        pooled = self.pool(conved).view(B, -1)
        latent = self.lin(pooled)
        return latent


class ClagnoscoDecoder(nn.Module):
    '''
    Вход: Латентные векторы в батче (B, 512)
    Выход: Восстановленные квадратные тензорные изображения в батче (B, 3, 256, 256)
    '''
    def __init__(self, latent_dim=512, image_res=256, negative_slope=0.01):
        super().__init__()
        self.init_size = image_res // 32

        self.lin = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*8),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.convt = nn.Sequential(
            nn.ConvTranspose2d(latent_dim*8, 256, kernel_size=self.init_size, stride=1),  # 1x1 → init_size x init_size
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, latent):
        """
        latent: [B, latent_dim]
        """
        x = torch.cat([latent], dim=1)                         # [B, latent_dim]
        lined = self.lin(x)                                    # [B, latent_dim * 16]
        img = self.convt(lined.unsqueeze(-1).unsqueeze(-1))    # [B, latent_dim * 16, 1, 1]
        return img                                             # [B, 3, 256, 256]


class ClagnoscoAutoencoder(nn.Module):
    '''
    Вход:
    - Тензорные изображения в батче (B, 3, H, W)

    Выход:
    - Латентные векторы в батче (B, 4096)
    - Восстановленные квадратные тензорные изображения в батче (B, 3, 256, 256)
    '''
    def __init__(
        self,
        latent_dim=512,
        backbone_channels=512,
        image_res=512,
        negative_slope=0.01
    ):
        super().__init__()
        self.encoder = ClagnoscoEncoder(
            latent_dim=latent_dim,
            backbone_channels=backbone_channels,
            negative_slope=negative_slope
        )
        self.decoder = ClagnoscoDecoder(
            latent_dim=latent_dim,
            image_res=image_res,
            negative_slope=negative_slope
        )

    def forward(self, x, decode=True):
        """
        x:     [B, 3, H, W] картинки
        """
        latent = self.encoder(x)
        if decode:
            recon = self.decoder(latent)
            return latent, recon
        else:
            return latent, None


def download_and_load_model(url, delete_temp=True):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Не удалось загрузить модель из {url}, код состояния: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    model_instance = ClagnoscoAutoencoder()
    model_instance.load_state_dict(torch.load(tmp_path))
    
    if delete_temp:
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"Не удалось удалить временный файл: {e}")

    print(f"Загружена модель из URL: {url}")
    return model_instance

def model_loader(model=None, first_epoch=0):
    '''
    - model: модель для загрузки (по умолчанию None, что загружает последнюю модель; создаёт новую модель при "create"; при директории загружает модель из файла; при URL загружает модель из интернета)
    - first_epoch: номер первой эпохи (по умолчанию 0, что означает первую эпоху)
    '''
    if model == "" or model is None:
        # Поиск последней модели в папке сохранений
        model_filenames = sorted([f for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')])
        model_filename = model_filenames[-1]
        first_epoch = int(model_filename.split('_')[4].split(".")[0])
        model = ClagnoscoAutoencoder()
        model.load_state_dict(torch.load(SAVE_FOLDER+model_filename))
        print(f"Загружена последняя модель: {model_filename}")

    elif type(model) == str:
        if model.lower() == "create":
            # Создание модели
            print("Создание новой модели")
            first_epoch = 0
            model = ClagnoscoAutoencoder()
        elif model.startswith("http"):
            # Загрузка модели из URL
            print(f"Загрузка модели из URL")
            model = download_and_load_model(model)
            first_epoch = -1
        else:
            # Загрузка модели из string
            first_epoch = int(model.split('_')[4].split(".")[0])
            model_filename = model
            model = ClagnoscoAutoencoder()
            model.load_state_dict(torch.load(SAVE_FOLDER+model_filename))
            print(f"Загружена выбранная модель: {model_filename}")
    return model, first_epoch


def train_autoencoder(transformed_dataset, train_batches, model=None,
                      num_epochs=10, first_epoch=0,
                      lr=1e-4):
    """
    Обучение модели автоэнкодера на преобразованном наборе данных.
    Модель сохраняется в папке SAVE_FOLDER ("./models/") с временной меткой.

    Входные данные:
        - transformed_dataset: экземпляр TransformedClagnoscoDataset
        - train_batches: список [(ширина, высота), [idx1, idx2, ...]] батчей
        - num_epochs: количество эпох для обучения (по умолчанию 10)
        - first_epoch: начальная эпоха для обучения (по умолчанию 0, что означает первую)
        - lr: скорость обучения для оптимизатора (по умолчанию 1e-4)
    
    Выходные данные (файлы сохраняются в SAVE_FOLDER):
        - model: обученная модель автоэнкодера
        - loss log: файл лога с значениями потерь для каждого шага
    """
    if first_epoch < 0:
        first_epoch = 0
    model, first_epoch = model_loader(model=model, first_epoch=first_epoch)
    model.train()
    model.to(DEVICE)

    # # Критерии потерь: MSE + SSIM
    # # MSELoss и StructuralSimilarityIndexMeasure из torchmetrics
    # mse_loss = nn.MSELoss()
    # ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # ssim_metric_weight = 1/16
    # mse_loss_weight = 1 - ssim_metric_weight

    # def criterion(pred, target):
    #     mse = mse_loss(pred, target)
    #     ssim = ssim_metric(pred, target)
    #     combined_loss = mse * mse_loss_weight + (1 - ssim) * ssim_metric_weight
    #     return combined_loss

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    len_train_batches = len(train_batches)

    for epoch in range(first_epoch, num_epochs):
        losses = []
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"autoencoder_{timestamp}_epoch_{epoch+1}.pt"
        loss_log_filename = f"autoencoder_{timestamp}_epoch_{epoch+1}_loss.txt"
        batched_buckets_gen = iterate_batched_buckets(transformed_dataset, train_batches)

        with open(SAVE_FOLDER+loss_log_filename, 'w') as loss_log:
            
            tqdm_bar = tqdm(range(len_train_batches), total=len_train_batches, desc=f"Эпоха {epoch+1}/{num_epochs}")
            for n, _ in enumerate(tqdm_bar):
                batch_dict, resolution = next(batched_buckets_gen)

                batch_imgs_inputs = batch_dict['image'].to(DEVICE)
                batch_imgs_targets = batch_dict['image_square'].to(DEVICE)

                optimizer.zero_grad()
                # Реконструкция картинок
                latent, recon = model(batch_imgs_inputs)
                loss_recon = criterion(recon, batch_imgs_targets)
                loss_recon.backward()
                optimizer.step()

                batch_loss = loss_recon.item()

                losses.append(batch_loss)
                loss_log.write(f"{batch_loss:.8f}\n")

                current_avg_loss = sum(losses) / len(losses)
                tqdm_bar.set_postfix(ср_потери=f"{current_avg_loss:.4f}", потери=f"{batch_loss:.4f}")

        avg_loss = sum(losses) / len(losses)
        torch.save(model.state_dict(), SAVE_FOLDER+model_filename)
        print(f"[Эпоха {epoch+1}] Средняя ошибка: {avg_loss:.6f} - Сохранено как {model_filename}")


def open_image(img):
    """
    Открыть изображение из URL или локального пути и преобразовать его в формат RGB.

    Аргументы:
        img: str (URL или локальный путь) или PIL.Image

    Возвращает:
        PIL.Image
    """
    if isinstance(img, str):
        if img.startswith("http"):
            return Image.open(requests.get(img, stream=True).raw).convert("RGB")
        else:
            return Image.open(img).convert("RGB")
    elif isinstance(img, Image.Image):
        return img.convert("RGB")
    else:
        raise ValueError("Input must be URL, local path, or PIL.Image")


def standardize_image_size(img, resize_size=192):
    """
    Привести изображение к стандартному размеру для модели.

    Аргументы:
        image: PIL.Image
        resize_size: int (по умолчанию 192 - минимальный размер стороны)

    Возвращает:
        PIL.Image
    """
    width, height = img.size
    if min(width, height) == resize_size:
        return img

    if width < height:
        new_width = resize_size
        new_height = int((resize_size / width) * height)
    else:
        new_height = resize_size
        new_width = int((resize_size / height) * width)
    new_size = new_width, new_height

    return img.resize(new_size, Image.LANCZOS)


def run_image_through_autoencoder(model, input_image, decode=True):
    """
    Пропустить изображение через автоэнкодер ClagnoscoAutoencoder и вернуть восстановленное изображение (PIL) и латентный вектор.

    Аргументы:
        input_image: PIL.Image или str (путь или URL)
        model: ClagnoscoAutoencoder
        decode: bool (по умолчанию True - восстанавливать изображение из латентного вектора)

    Возвращает:
        restored_pil (PIL.Image), latent (Tensor), embedding (Tensor)
    """
    input_image = standardize_image_size(open_image(input_image))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        latent, recon = model(input_tensor, decode=decode)

    if decode:
        restored_tensor = recon.squeeze(0).cpu().clamp(0, 1)  # [3, H, W]
        restored_pil = transforms.ToPILImage()(restored_tensor)

        return latent.squeeze(0), restored_pil
    else:
        return latent.squeeze(0), None


def test_model(model, test_batches, transformed_dataset):
    """
    Проверка модели на тестовом наборе данных.

    Входные данные:
        - model: обученная модель автоэнкодера
        - test_batches: список [(ширина, высота), [idx1, idx2, ...]] батчей
        - transformed_dataset: экземпляр TransformedClagnoscoDataset
    Выходные данные:
        - avg_loss: средняя ошибка пиксельной реконструкции (MSE) для каждого тестового изображения
        - losses: ошибка пиксельной реконструкции (MSE) для каждого тестового изображения
    """
    model.eval()
    model.to(DEVICE)

    test_list = batch_buckets_to_list(test_batches)
    losses = []
    tqdm_bar = tqdm(test_list, desc="Проверка модели")
    for test_idx in tqdm_bar:
        sample = transformed_dataset[test_idx]
        test_image = sample['image'].unsqueeze(0).to(DEVICE)
        test_image_square = sample['image_square'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            latent, recon = model(test_image)
            loss = F.mse_loss(recon, test_image_square).item()
        losses.append(loss)
        current_avg_loss = sum(losses) / len(losses)
        tqdm_bar.set_postfix(ср_потери=f"{current_avg_loss:.4f}", потери=f"{loss:.4f}")
    avg_loss = sum(losses) / len(losses)
    print(f"Средняя ошибка: {avg_loss:.6f}")
    return avg_loss, losses


def test_models(model_list, test_batches, transformed_dataset, save_avg=True):
    """
    Тестирование нескольких моделей автоэнкодера на преобразованном наборе данных.

    Входные данные:
        - model_list: список названий моделей (строки) для тестирования
        - test_batches: список [(ширина, высота), [idx1, idx2, ...]] батчей
        - transformed_dataset: экземпляр TransformedClagnoscoDataset
    
    Выходные данные:
        - avg_losses_dict: словарь с названиями моделей в качестве ключей и средней ошибкой пиксельной реконструкции (MSE) в качестве значений
        - losses_dict: словарь с названиями моделей в качестве ключей и списками ошибок пиксельной реконструкции (MSE) в качестве значений
    """
    avg_losses_dict = {}
    losses_dict = {}
    for model in model_list:
        print(f"Проверка модели: {model}")
        model_instance, _ = model_loader(model=model)
        avg_losses, losses = test_model(model_instance, test_batches, transformed_dataset)
        avg_losses_dict[model] = avg_losses
        losses_dict[model] = losses
    if save_avg:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        avg_losses_filename = f"autoencoder_{timestamp}_avg_losses.txt"
        with open(SAVE_FOLDER+avg_losses_filename, 'w') as avg_loss_log:
            for model, avg_loss in avg_losses_dict.items():
                avg_loss_log.write(f"{model}: {avg_loss:.8f}\n")
    return avg_losses_dict, losses_dict


def delete_untrained_loss_log_files():
    """
    Удаление файлов, которые не соответствуют ни одной модели в папке сохранений.
    """
    models = sorted([f.split('.pt')[0] for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')])
    for filename in sorted([f for f in os.listdir(SAVE_FOLDER) if f.endswith('_loss.txt')])[:-1]:
        if filename.split('_loss.txt')[0] not in models:
            file_path = os.path.join(SAVE_FOLDER, filename)
            try:
                os.remove(file_path)
                print(f"Удалён файл лога потерь: {file_path}")
            except Exception as e:
                print(f"Ошибка при удалении файла {file_path}: {e}")


def images_to_latent(image_folder, model=None, cashing=False):
    """
    Преобразование изображений из папки в латентные векторы с помощью модели автоэнкодера.
    Аргументы:
        image_folder: str (путь к папке с изображениями)
        model: ClagnoscoAutoencoder или str (сама модель, путь к модели или URL)
        cashing: bool (по умолчанию False) - кэшировать и использовать латентные векторы в файлах _latent.npy)
    Возвращает:
        images_and_latents: список кортежей (путь к изображению, латентный вектор)
        errored_images: список изображений, которых не получилось обработать
    """
    
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {image_folder}")
    
    model, _ = model_loader(model=model)
    model.to(DEVICE)

    images_and_latents = []
    errored_images = []
    for filename in os.listdir(image_folder):
        try:
            if cashing:
                latent_path = os.path.join(image_folder, filename + "_latent.npy")
                if os.path.exists(latent_path):
                    latent = np.load(latent_path)
                    images_and_latents.append((filename, latent))
                    continue
            image_path = os.path.join(image_folder, filename)
            latent, _ = run_image_through_autoencoder(model, image_path, decode=False)
            images_and_latents.append((filename, latent.cpu().numpy()))
            if cashing:
                np.save(latent_path, latent.cpu().numpy())
        except Exception as e:
            if not filename.endswith("_latent.npy"):
                errored_images.append(filename)
    
    return images_and_latents, errored_images

def clear_cash(image_folder):
    """
    Очистка кэша латентных векторов в папке с изображениями.
    Аргументы:
        image_folder: str (путь к папке с изображениями)
    Возвращает:
        count: int - количество удалённых файлов кэша
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {image_folder}")
    count = 0
    for filename in os.listdir(image_folder):
        if filename.endswith("_latent.npy"):
            os.remove(os.path.join(image_folder, filename))
            count += 1
    return count
