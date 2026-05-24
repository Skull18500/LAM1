# extract_latents.py – Encode all frame pairs with the AE and save latents
import torch
import cv2
from safetensors.torch import load_model
from tiny_lam import GlobalLAM   # your model class (encoder+decoder, codebook ignored)

# ---------------- settings ----------------
VIDEO_PATH = "dataset1.mp4"
AE_CHECKPOINT = "lam_ae_pretrained.safetensors"   # the autoencoder weights
OUTPUT_FILE = "latents.pt"
TARGET_FPS = 6
FRAME_SIZE = (128, 128)
BATCH_SIZE = 1           # process one pair at a time to save RAM
LATENT_DIM = 32
# ------------------------------------------

# Reload the same frame extraction logic (no dataset class needed)
def extract_frames(video_path, target_fps, frame_size):
    frames = []
    cap = cv2.VideoCapture(video_path)
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
            frames.append(frame)
    cap.release()
    return frames

print("Extracting frames...")
frames = extract_frames(VIDEO_PATH, TARGET_FPS, FRAME_SIZE)
print(f"Extracted {len(frames)} frames.")

# Load AE
model = GlobalLAM(codebook_size=12, latent_dim=LATENT_DIM)
load_model(model, AE_CHECKPOINT)
model.eval()

latents = []

print("Encoding frame pairs...")
with torch.no_grad():
    for i in range(len(frames) - 1):
        f_t = frames[i].unsqueeze(0)      # [1, 3, H, W]
        f_t1 = frames[i+1].unsqueeze(0)
        diff = f_t1 - f_t
        z_e = model.encode(diff)          # [1, latent_dim, 1, 1]
        z_e = z_e.squeeze()               # [latent_dim]
        latents.append(z_e)

        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(frames)-1}")

# Stack into a single tensor [num_pairs, latent_dim]
latent_tensor = torch.stack(latents)
print(f"Latent tensor shape: {latent_tensor.shape}")

# Save
torch.save(latent_tensor, OUTPUT_FILE)
print(f"Saved {len(latents)} latents to {OUTPUT_FILE}")