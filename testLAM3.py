import torch, cv2
from safetensors.torch import load_model
from tiny_lam import GlobalLAM 
import numpy as np
from PIL import Image

# --- Paths & Settings ---
IMAGE_PATH = "download1.png"
AE_PATH = "lam_pretrained.safetensors"

OUTPUT_FPS = 100       # 100 FPS for a fast timelapse
FRAME_SIZE = (128, 128)
DREAM_STEPS = 10000    # Running for a massive 10k frames
NOISE_INJECTION = 0.05 

# --- Load Model ---
ae = GlobalLAM(codebook_size=12, latent_dim=32)
load_model(ae, AE_PATH)
ae.eval()

def load_start_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((FRAME_SIZE[1], FRAME_SIZE[0]))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

try:
    base_img = load_start_image(IMAGE_PATH)
except Exception as e:
    print(f"Could not load {IMAGE_PATH}: {e}")
    exit()

print(f"Starting 10,000-frame hallucination test on {IMAGE_PATH}...")

prev = base_img.clone()
curr = base_img.clone()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("10k_ae_test.mp4", fourcc, OUTPUT_FPS, (FRAME_SIZE[1], FRAME_SIZE[0]))

out_img = (curr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
writer.write(out_img)

with torch.no_grad():
    for t in range(DREAM_STEPS):
        f_t = prev.unsqueeze(0)
        f_t1 = curr.unsqueeze(0)
        
        diff = f_t1 - f_t
        diff = diff + NOISE_INJECTION * torch.randn_like(diff)

        # --- PURE CONTINUOUS AE ---
        z_e = ae.encode(diff)         
        recon_diff = ae.decode(z_e)   

        # --- Predict Next Frame ---
        next_frame = f_t1 + recon_diff
        next_frame = torch.clamp(next_frame, 0.0, 1.0).squeeze(0)

        out_img = (next_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        writer.write(out_img)

        prev, curr = curr, next_frame

        if (t + 1) % 500 == 0:
            print(f"Hallucinated frame {t + 1}/{DREAM_STEPS}")

writer.release()

# --- SAVE THE FINAL FRAME ---
cv2.imwrite("finalframe.png", out_img)
print("Test complete! Saved 10k_ae_test.mp4 and finalframe.png")