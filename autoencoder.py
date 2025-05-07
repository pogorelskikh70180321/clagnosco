import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
if __name__ == "__main__":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

from dataset import *

SAVE_FOLDER = "models/"


class ClagnoscoEncoder(nn.Module):
    '''
    Input: resized image

    Output: latent values, embedding
    '''
    def __init__(
        self,
        latent_dim=512,
        embed_dim=768,
        backbone_channels=2048,
        hidden_dim=1536,
        negative_slope=0.01
    ):
        super().__init__()
        self.latent_dim_size = latent_dim
        self.embed_dim_size = embed_dim

        self.conv = nn.Sequential(
            # Conv Layer 1: kernel_size=4, stride=2, padding=1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv Layer 2: kernel_size=4, stride=2, padding=1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv Layer 3: kernel_size=3, stride=1, padding=1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv Layer 4: kernel_size=3, stride=1, padding=1
            nn.Conv2d(in_channels=256, out_channels=backbone_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(backbone_channels, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.fc2 = nn.Linear(hidden_dim, latent_dim + embed_dim)

    def forward(self, x):
        B = x.size(0)
        feat = self.conv(x)
        pooled = self.pool(feat).view(B, -1)
        hidden = self.act(self.fc1(pooled))
        out = self.fc2(hidden)
        latent, emb_pred = out.split([self.latent_dim_size, self.embed_dim_size], dim=1)
        return latent, emb_pred

class ClagnoscoDecoder(nn.Module):
    '''
    Input: latent values, embedding, ratio

    Output: restored square image
    '''
    def __init__(
        self,
        latent_dim=512,
        embed_dim=768,
        image_res=512,
        negative_slope=0.01
    ):
        super().__init__()
        self.input_dim = latent_dim + embed_dim + 1  # +1 for ratio
        self.init_size = image_res // 32

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256 * self.init_size * self.init_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Unflatten(1, (256, self.init_size, self.init_size)),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(in_channels=64,  out_channels=32,  kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(in_channels=32,  out_channels=3,   kernel_size=4, stride=2, padding=1),  # 128 → 256
            nn.Sigmoid()
        )

    def forward(self, latent, emb_pred, ratio):
        """
        latent: [B, latent_dim]
        emb_pred:  [B, embed_dim]
        ratio:  [B, 1] (e.g., torch.tensor([[1.0], [1.33], ...]))
        """
        x = torch.cat([latent, emb_pred, ratio], dim=1)
        x = self.fc(x)
        x = self.deconv(x)
        return x

class ClagnoscoAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        embed_dim=768,
        backbone_channels=2048,
        hidden_dim=1536,
        image_res=512,
        negative_slope=0.01
    ):
        super().__init__()
        self.encoder = ClagnoscoEncoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            backbone_channels=backbone_channels,
            hidden_dim=hidden_dim,
            negative_slope=negative_slope
        )
        self.decoder = ClagnoscoDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            image_res=image_res,
            negative_slope=negative_slope
        )

    def forward(self, x, ratio):
        """
        x:     [B, 3, H, W]
        ratio: [B, 1] (e.g., torch.tensor([[1.0], [1.33], ...]))
        """
        latent, emb_pred = self.encoder(x)
        recon = self.decoder(latent, emb_pred, ratio)
        return latent, emb_pred, recon


def train_autoencoder(transformed_dataset, train_batches, model=None,
                      num_epochs=10, first_epoch=0,
                      lr=1e-4, loss_recon_weight=0.5):
    """
    Train the autoencoder model on the transformed dataset.
    The model is saved in the SAVE_FOLDER ("./models/") with a timestamp.

    Inputs:
        - transformed_dataset: instance of TransformedClagnoscoDataset
        - train_batches: list of [(width, height), [idx1, idx2, ...]] batches
        - model: name of the model to load (default is None, which loads the latest model; create a new model if "create")
        - num_epochs: number of epochs to train (default is 10)
        - first_epoch: starting epoch for training (default is 0, meaning first)
        - lr: learning rate for the optimizer (default is 1e-4)
        - loss_recon_weight: weight for reconstruction loss vs embedding loss (default is 0.5)
    Outputs (files saved in SAVE_FOLDER):
        - model: trained autoencoder model
        - loss log: log file with loss values for each step
    """

    if model == "" or model is None:
        # Find latest model in the models folder
        model_filenames = sorted([f for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')])
        first_epoch = int(model_filenames[-1].split('_')[4].split(".")[0])
        model = ClagnoscoAutoencoder()
        model.load_state_dict(torch.load(SAVE_FOLDER+model_filenames[-1]))
    elif type(model) == str:
        if model.lower() == "create":
            # Create a new model
            first_epoch = 0
            model = ClagnoscoAutoencoder()
        else:
            # Load model name from string
            first_epoch = int(model.split('_')[4].split(".")[0])
            model_filename = model
            model = ClagnoscoAutoencoder()
            model.load_state_dict(torch.load(SAVE_FOLDER+model_filename))
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
            
            for _ in tqdm(range(len_train_batches), total=len_train_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_dict, batch_ratios, resolution = next(batched_buckets_gen)

                batch_imgs_inputs = batch_dict['image'].to(DEVICE)
                batch_imgs_targets = batch_dict['image_square'].to(DEVICE)
                batch_embeddings = batch_dict['embedding'].to(DEVICE)
                batch_ratios = batch_ratios.to(DEVICE)

                optimizer.zero_grad()
                
                latent, emb_pred, recon = model(batch_imgs_inputs, batch_ratios)

                loss_emb_pred = criterion(emb_pred, batch_embeddings)
                loss_recon = criterion(recon, batch_imgs_targets)

                loss = (1 - loss_recon_weight) * loss_emb_pred + loss_recon_weight * loss_recon
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                losses.append(batch_loss)
                loss_log.write(f"{batch_loss:.8f}\n")

        avg_loss = sum(losses) / len(losses)
        torch.save(model.state_dict(), SAVE_FOLDER+model_filename)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f} - Saved to {model_filename}")

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