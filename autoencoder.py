import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import requests
if __name__ == "__main__":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

from dataset import *

SAVE_FOLDER = "models/"


class ClagnoscoEncoder(nn.Module):
    '''
    Input: resized image tensor of shape (B, 3, H, W)
    Output: latent vector of size latent_dim (1024)
    '''
    def __init__(
        self,
        latent_dim=1024,
        backbone_channels=1024,
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

        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # Output: (B, backbone_channels, 2, 2)

        self.lin = nn.Sequential(
            nn.Linear(backbone_channels * 2 * 2, latent_dim),
        )

    def forward(self, x):
        B = x.size(0)
        conved = self.conv(x)
        pooled = self.pool(conved).view(B, -1)
        latent = self.lin(pooled)
        return latent


class ClagnoscoDecoder(nn.Module):
    '''
    Input: latent values, ratio
    Output: restored square image
    '''
    def __init__(self, latent_dim=1024, image_res=256, negative_slope=0.01):
        super().__init__()
        self.init_size = image_res // 32

        self.lin = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=self.init_size, stride=1),  # 1×1 → init_size×init_size
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16→32
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32→64
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 64→128
            nn.Sigmoid()
        )

    def forward(self, latent, ratio):
        """
        latent: [B, latent_dim]
        ratio:  [B, 1]
        """
        x = torch.cat([latent, ratio], dim=1)                  # [B, latent_dim + 1]
        lined = self.lin(x)                                    # [B, latent_dim]
        img = self.decoder(lined.unsqueeze(-1).unsqueeze(-1))  # [B, latent_dim, 1, 1]
        return img                                             # [B, 3, 256, 256]


class ClagnoscoAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim=1024,
        backbone_channels=1024,
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

    def forward(self, x, ratio):
        """
        x:     [B, 3, H, W]
        ratio: [B, 1] (e.g., torch.tensor([[1.0], [1.33], ...]))
        """
        # latent, emb_pred = self.encoder(x)
        # recon = self.decoder(latent, emb_pred, ratio)
        latent = self.encoder(x)
        recon = self.decoder(latent, ratio)
        return latent, recon


def model_loader(model=None, first_epoch=0):
    '''
    - model: name of the model to load (default is None, which loads the latest model; create a new model if "create")
    '''
    if model == "" or model is None:
        # Find latest model in the models folder
        model_filenames = sorted([f for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')])
        model_filename = model_filenames[-1]
        first_epoch = int(model_filename.split('_')[4].split(".")[0])
        model = ClagnoscoAutoencoder()
        model.load_state_dict(torch.load(SAVE_FOLDER+model_filename))
        print(f"Loading latest model: {model_filename}")
    elif type(model) == str:
        if model.lower() == "create":
            # Create a new model
            print("Creating a new model")
            first_epoch = 0
            model = ClagnoscoAutoencoder()
        else:
            # Load model name from string
            first_epoch = int(model.split('_')[4].split(".")[0])
            model_filename = model
            model = ClagnoscoAutoencoder()
            model.load_state_dict(torch.load(SAVE_FOLDER+model_filename))
            print(f"Loading selected model: {model_filename}")
    return model, first_epoch

def train_autoencoder(transformed_dataset, train_batches, model=None,
                      num_epochs=10, first_epoch=0,
                      lr=1e-4, loss_recon_weight=1.0):
    """
    Train the autoencoder model on the transformed dataset.
    The model is saved in the SAVE_FOLDER ("./models/") with a timestamp.

    Inputs:
        - transformed_dataset: instance of TransformedClagnoscoDataset
        - train_batches: list of [(width, height), [idx1, idx2, ...]] batches
        - num_epochs: number of epochs to train (default is 10)
        - first_epoch: starting epoch for training (default is 0, meaning first)
        - lr: learning rate for the optimizer (default is 1e-4)
        # - loss_recon_weight: weight for reconstruction loss vs embedding loss (default is 0.5)
    Outputs (files saved in SAVE_FOLDER):
        - model: trained autoencoder model
        - loss log: log file with loss values for each step
    """
    model, first_epoch = model_loader(model=model, first_epoch=first_epoch)
    model.train()
    model.to(DEVICE)

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
            
            tqdm_bar = tqdm(range(len_train_batches), total=len_train_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
            for n, _ in enumerate(tqdm_bar):
                batch_dict, batch_ratios, resolution = next(batched_buckets_gen)

                batch_imgs_inputs = batch_dict['image'].to(DEVICE)
                batch_imgs_targets = batch_dict['image_square'].to(DEVICE)
                # batch_embeddings = batch_dict['embedding'].to(DEVICE)
                batch_ratios = batch_ratios.to(DEVICE)

                # optimizer.zero_grad()
                # latent, emb_pred, recon = model(batch_imgs_inputs, batch_ratios)
                # # step 1: enbedding
                # loss_emb_pred = criterion(emb_pred, batch_embeddings) * (1 - loss_recon_weight)
                # loss_emb_pred.backward()
                # optimizer.step()

                optimizer.zero_grad()
                # latent, emb_pred, recon = model(batch_imgs_inputs, batch_ratios)
                latent, recon = model(batch_imgs_inputs, batch_ratios)
                # step 2: reconstruction
                loss_recon = criterion(recon, batch_imgs_targets) # * loss_recon_weight
                loss_recon.backward()
                optimizer.step()

                # loss = loss_emb_pred + loss_recon
                loss = loss_recon
                batch_loss = loss.item()
                # batch_loss_emb_pred = loss_emb_pred.item()
                # batch_loss_recon = loss_recon.item()

                # tqdm_bar.set_postfix(loss=f"{batch_loss:.4f}", loss_emb_pred=f"{batch_loss_emb_pred:.4f}", loss_recon=f"{batch_loss_recon:.4f}")
                tqdm_bar.set_postfix(loss=f"{batch_loss:.4f}")

                losses.append(batch_loss)
                # loss_log.write(f"{batch_loss:.8f}\t{batch_loss_emb_pred:.8f}\t{batch_loss_recon:.8f}\n")
                loss_log.write(f"{batch_loss:.8f}\n")

        avg_loss = sum(losses) / len(losses)
        torch.save(model.state_dict(), SAVE_FOLDER+model_filename)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f} - Saved to {model_filename}")

def run_image_through_autoencoder(model, input_image):
    """
    Pass an image through the ClagnoscoAutoencoder and return the restored image (PIL), embedding and latent.

    Args:
        input_image: PIL.Image or str (path or URL)
        model: ClagnoscoAutoencoder (must be in eval mode)

    Returns:
        restored_pil (PIL.Image), latent (Tensor), embedding (Tensor)
    """
    if isinstance(input_image, str):
        if input_image.startswith("http"):
            input_image = Image.open(requests.get(input_image, stream=True).raw).convert("RGB")
        else:
            input_image = Image.open(input_image).convert("RGB")

    w, h = input_image.size
    ratio = torch.tensor([[w / h]], dtype=torch.float32).to(DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    with torch.no_grad():
        # latent, emb_pred, recon = model(input_tensor, ratio)
        latent, recon = model(input_tensor, ratio)

    restored_tensor = recon.squeeze(0).cpu().clamp(0, 1)  # [3, H, W]
    restored_pil = transforms.ToPILImage()(restored_tensor)

    # return latent.squeeze(0), emb_pred.squeeze(0), restored_pil
    return latent.squeeze(0), restored_pil

def delete_untrained_loss_log_files():
    """
    Delete untrained loss log files in the SAVE_FOLDER.
    """
    models = sorted([f.split('.pt')[0] for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')])
    for filename in sorted([f for f in os.listdir(SAVE_FOLDER) if f.endswith('_loss.txt')])[:-1]:
        if filename.split('_loss.txt')[0] not in models:
            file_path = os.path.join(SAVE_FOLDER, filename)
            try:
                os.remove(file_path)
                print(f"Deleted untrained log loss file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
