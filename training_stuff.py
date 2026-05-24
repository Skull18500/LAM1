import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SpacedFrameDataset(Dataset):
    def __init__(self, video_path, target_fps=6, input_len=4, spacing=6,
                 frame_size=(128, 128), cache_frames=True):
        """
        Args:
            video_path: path to the video file
            target_fps: frames per second to sample from the video
            input_len: number of input frames in each sequence
            spacing: number of frames between each input frame (at target_fps)
            frame_size: (height, width) to resize frames to
            cache_frames: if True, preload all frames into memory (faster but uses RAM)
        """
        self.video_path = video_path
        self.input_len = input_len
        self.spacing = spacing
        self.frame_size = frame_size

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = original_fps / target_fps  # number of original frames per target frame

        # Pre-compute the indices (in original frame space) we need
        self.frame_indices = []
        for i in range(total_frames):
            # if the current original frame index corresponds to a target fps time point
            if i % step < 1.0:  # simple check: take the frame closest to the time point
                self.frame_indices.append(i)
        # Alternative, more precise: we could sample at exact time points using cap.set(cv2.CAP_PROP_POS_MSEC)
        # but this approximate method is faster.

        if cache_frames:
            self.frames = []
            for idx in self.frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    # If we can't read, just duplicate the last frame (or break)
                    if self.frames:
                        self.frames.append(self.frames[-1])
                    else:
                        self.frames.append(torch.zeros(3, *frame_size))
                    continue
                frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [0,1]
                self.frames.append(frame)
            cap.release()
        else:
            # If not caching, we'll read on the fly (slower)
            self.frames = None  # We'll use cap in __getitem__ (not implemented here)
            cap.release()  # We'll re-open in __getitem__ if needed; but for simplicity we'll cache.

        # Number of possible sequences
        # Each sequence requires (input_len-1)*spacing frames before the last input, plus one target frame
        self.valid_starts = len(self.frames) - (input_len - 1) * spacing - 1

    def __len__(self):
        return max(0, self.valid_starts)

    def __getitem__(self, idx):
        # Build input sequence of length input_len, spaced by spacing
        input_frames = []
        for k in range(self.input_len):
            frame_idx = idx + k * self.spacing
            input_frames.append(self.frames[frame_idx])
        # Target is the very next frame after the last input
        target_idx = idx + self.input_len * self.spacing
        target = self.frames[target_idx]

        # Stack input frames: [input_len, C, H, W] -> [input_len*C, H, W]
        input_tensor = torch.cat(input_frames, dim=0)  # [input_len*3, H, W]
        return input_tensor, target