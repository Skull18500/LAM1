# dream_ae_noise_test.py – Triangle AE with variable noise
import torch, cv2
from safetensors.torch import load_model
from tiny_lam import GlobalLAM
import numpy as np

# Paths
VIDEO_PATH = "dataset1.mp4"
AE_PATH = "lam_pretrained.safetensors"       # your triangle‑AE

# Settings
TRAIN_FPS = 6
OUTPUT_FPS = 100
FRAME_SIZE = (128, 128)
DREAM_STEPS = 10000
START_FRAME = 120
NOISE_LEVELS = [0.2]

# ---------------- Load AE ----------------
ae = GlobalLAM(codebook_size=12, latent_dim=32)
load_model(ae, AE_PATH)
ae.eval()

# ---------------- Extract frames ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(orig_fps / TRAIN_FPS))

def read_target_frame():
    for _ in range(step):
        ret, frame = cap.read()
        if not ret: return None
    frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]))
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return frame

# Skip to START_FRAME
for _ in range(START_FRAME * step):
    cap.read()

prev = read_target_frame()
curr = read_target_frame()
cap.release()
if prev is None or curr is None:
    print("Could not read starting frames.")
    exit()

# ---------------- Generate one video per noise level ----------------
for noise_std in NOISE_LEVELS:
    print(f"Testing noise std = {noise_std}")
    p, c = prev.clone(), curr.clone()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"kadath_ae_triangle_noise{noise_std:.3f}.mp4",
                             fourcc, OUTPUT_FPS,
                             (FRAME_SIZE[1], FRAME_SIZE[0]))

    with torch.no_grad():
        for t in range(DREAM_STEPS):
            f_t = p.unsqueeze(0)
            f_t1 = c.unsqueeze(0)
            diff = f_t1 - f_t

            # Inject Gaussian noise into the difference
            diff = diff + noise_std * torch.randn_like(diff)

            z = ae.encode(diff)
            recon_diff = ae.decode(z)

            next_frame = c.unsqueeze(0) + recon_diff
            next_frame = torch.clamp(next_frame, 0.0, 1.0).squeeze(0)

            out_img = (next_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            writer.write(out_img)

            p, c = c, next_frame  # slide window

    writer.release()
    print(f"Saved kadath_ae_triangle_noise{noise_std:.3f}.mp4")