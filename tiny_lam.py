# tiny_lam.py – Complete retrained LAM with autoencoder pretraining
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from safetensors.torch import save_model, load_model
from tqdm import tqdm
from random_stuff import LightAngleRotate, TriangleDistActivation

# ------------------------------------------------------------
# 1. TinyLAM (unchanged architecture, but training logic fixed)
# ------------------------------------------------------------
class GlobalLAM(nn.Module):
    def __init__(self, codebook_size=12, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  nn.SiLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, latent_dim, 4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.codebook = nn.Embedding(codebook_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 33, 4, stride=2, padding=1),  # 1 → 2
            TriangleDistActivation(33),  # custom activation to mix channels
            nn.ConvTranspose2d(33, 16, 4, stride=2, padding=1),          # 2 → 4
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),           # 4 → 8
            nn.SiLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),            # 8 → 16
            nn.SiLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 16 → 128
            nn.Conv2d(4, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, diff):
        return self.encoder(diff)

    def decode(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), -1, 1, 1)
        return self.decoder(z)

    def forward(self, frame_t, frame_t1):
        diff = frame_t1 - frame_t
        z_e = self.encoder(diff)                     # [B, latent_dim, 1, 1]
        z_e = z_e.squeeze(-1).squeeze(-1)            # [B, latent_dim]

        dist = torch.cdist(z_e, self.codebook.weight)
        idx = dist.argmin(dim=-1)
        z_q = self.codebook(idx)

        z_q_ste = z_e + (z_q - z_e).detach()
        recon = self.decoder(z_q_ste.view(-1, self.latent_dim, 1, 1))
        return idx, z_q_ste, z_q, z_e, recon

# -----------------------------------
# 2. Dataset: consecutive frame pairs
# -----------------------------------
class FramePairDataset(Dataset):
    def __init__(self, video_path, target_fps=6, frame_size=(128, 128)):
        self.frame_size = frame_size
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(orig_fps / target_fps))
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                self.frames.append(frame)
        cap.release()
        print(f"Extracted {len(self.frames)} frames from video.")

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, idx):
        return self.frames[idx], self.frames[idx + 1]


# -----------------------------------------------
# 3. Two‑stage training function
# -----------------------------------------------
def train_lam(video_path, output_path="lam_pretrained.safetensors",
              codebook_size=8, latent_dim=32, target_fps=6,
              batch_size=8, lr=1e-3, epochs_ae=5, epochs_vq=15, device="cpu"):

    dataset = FramePairDataset(video_path, target_fps=target_fps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GlobalLAM(codebook_size=codebook_size, latent_dim=latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ---- Stage 1: Autoencoder pretraining (no codebook) ----
    print("=== Stage 1: Autoencoder pretraining ===")
    for epoch in range(epochs_ae):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"AE Epoch {epoch+1}/{epochs_ae}")
        for frame_t, frame_t1 in pbar:
            frame_t, frame_t1 = frame_t.to(device), frame_t1.to(device)
            diff = frame_t1 - frame_t
            z = model.encode(diff)          # skip quantisation
            recon = model.decode(z)
            loss = loss_fn(recon, diff)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"AE Epoch {epoch+1} | Avg Loss: {epoch_loss/len(loader):.4f}")

    # Save the pretrained autoencoder weights (optional)
    save_model(model, "lam_ae_pretrained.safetensors")
    print("Autoencoder checkpoint saved.")

    # ---- Stage 2: VQ‑VAE training with codebook ----
    print("\n=== Stage 2: VQ‑VAE training ===")
    commitment_weight = 0.25   # lower weight initially, can increase later
    codebook_weight = 1.0

    for epoch in range(epochs_vq):
        epoch_recon = 0.0
        epoch_commit = 0.0
        epoch_codebook = 0.0
        usage = torch.zeros(codebook_size, dtype=torch.long)

        pbar = tqdm(loader, desc=f"VQ Epoch {epoch+1}/{epochs_vq}")
        for frame_t, frame_t1 in pbar:
            frame_t, frame_t1 = frame_t.to(device), frame_t1.to(device)

            idx, z_q_ste, z_q, z_e, recon = model(frame_t, frame_t1)
            target_diff = frame_t1 - frame_t

            loss_recon = loss_fn(recon, target_diff)
            loss_commit = loss_fn(z_q, z_e.detach())
            loss_codebook = loss_fn(z_q.detach(), z_e)

            total = loss_recon + commitment_weight * loss_commit + codebook_weight * loss_codebook

            optim.zero_grad()
            total.backward()
            optim.step()

            usage += torch.bincount(idx.cpu(), minlength=codebook_size)

            epoch_recon += loss_recon.item()
            epoch_commit += loss_commit.item()
            epoch_codebook += loss_codebook.item()
            pbar.set_postfix(recon=f"{loss_recon.item():.4f}",
                             commit=f"{loss_commit.item():.4f}",
                             codebook=f"{loss_codebook.item():.4f}")

        # Codebook reset for unused codes
        unused = (usage == 0).nonzero(as_tuple=True)[0]
        if len(unused) > 0:
            # For global LAM, z_e is already [B, latent_dim] - just flatten the batch
            z_e_flat = z_e.detach().cpu()  # [B, latent_dim]
            # If you were accumulating z_e across batches, you'd need a buffer.
            # For simplicity, we just use the last batch's z_e.
            # To make it more robust, reshape to [N, latent_dim]
            rand_idx = torch.randint(0, z_e_flat.shape[0], (len(unused),))
            replacements = z_e_flat[rand_idx].to(device)
            model.codebook.weight.data[unused] = replacements
            print(f"  Reset {len(unused)} unused codes: {unused.tolist()}")

        used_codes = (usage > 0).sum().item()
        print(f"Epoch {epoch+1} | Recon {epoch_recon/len(loader):.4f} "
              f"Commit {epoch_commit/len(loader):.4f} Codebook {epoch_codebook/len(loader):.4f} "
              f"Codes used: {used_codes}/{codebook_size}")

        if (epoch + 1) % 5 == 0:
            ckpt = f"lam_epoch{epoch+1}.safetensors"
            save_model(model, ckpt)
            print(f"Checkpoint saved to {ckpt}")

    save_model(model, output_path)
    print(f"Final LAM saved to {output_path}")


# -----------------
# 4. Main entry
# -----------------
if __name__ == "__main__":
    VIDEO_PATH = "dataset1.mp4"   # change to your video

    train_lam(
        video_path=VIDEO_PATH,
        output_path="lam_pretrained.safetensors",
        codebook_size=12,
        latent_dim=32,
        target_fps=6,
        batch_size=8,
        lr=1e-3,
        epochs_ae=2,        # autoencoder pretraining epochs
        epochs_vq=8,       # VQ‑VAE epochs after pretraining
        device="cpu"
    )