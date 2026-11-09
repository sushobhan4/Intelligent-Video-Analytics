"""
================================================================================
Neural CCTV Analyser - Anchor-Based Temporal Motion Gating
================================================================================
Implements 10-frame anchor differencing to eliminate 90%+ of GPU idle computation.
Divides a 30 FPS video into 3 chunks per second. Computes pixel-wise absolute
differences against an anchor frame (Frame 0 of each chunk). Only when the mean
difference crosses the threshold is the chunk dispatched to the neural pipeline.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class MotionGate:
    """
    Anchor-based temporal chunk motion detector with optional zone-masking.
    """

    def __init__(
        self,
        chunk_size: int = 10,
        motion_threshold: float = 4.0,
        downsample_size: Tuple[int, int] = (320, 180),
        blur_kernel: Tuple[int, int] = (5, 5),
        zone_mask_polygon: Optional[List[Tuple[float, float]]] = None
    ):
        self.chunk_size = chunk_size
        self.motion_threshold = motion_threshold
        self.downsample_size = downsample_size
        self.blur_kernel = blur_kernel
        self.zone_mask_polygon = zone_mask_polygon
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_mask_area: float = float(downsample_size[0] * downsample_size[1])

        if zone_mask_polygon:
            self._build_zone_mask(zone_mask_polygon)

    def _build_zone_mask(self, polygon: List[Tuple[float, float]]):
        """Constructs a binary mask normalized to the downsampled resolution."""
        w, h = self.downsample_size
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[int(px * w), int(py * h)] for px, py in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self._cached_mask = mask > 0
        mask_area = np.count_nonzero(self._cached_mask)
        self._cached_mask_area = float(max(mask_area, 1))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Converts frame to small blurred grayscale representation."""
        small = cv2.resize(frame, self.downsample_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel[0] > 0:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return gray

    def evaluate_chunk(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int]
    ) -> Tuple[bool, float, float, int, np.ndarray]:
        """
        Evaluates a chunk of frames (typically 10 frames).
        
        Returns:
            is_active (bool): True if chunk motion exceeds threshold.
            mean_diff (float): Average pixel delta across all frames vs anchor.
            max_diff (float): Maximum single-frame delta in the chunk.
            peak_frame_idx (int): Global frame index with the highest activity.
            peak_frame (np.ndarray): Original full-resolution image of the peak frame.
        """
        if not frames:
            return False, 0.0, 0.0, 0, np.zeros((10, 10, 3), dtype=np.uint8)

        # Frame 0 is Anchor Frame
        anchor_gray = self._preprocess(frames[0])

        if len(frames) == 1:
            return True, self.motion_threshold + 1.0, self.motion_threshold + 1.0, frame_indices[0], frames[0]

        deltas = []
        max_delta = -1.0
        max_idx_in_chunk = 0

        for i in range(1, len(frames)):
            cur_gray = self._preprocess(frames[i])
            diff = cv2.absdiff(cur_gray, anchor_gray)

            if self._cached_mask is not None:
                # Calculate mean difference only inside the polygon of interest
                val = float(np.sum(diff[self._cached_mask])) / self._cached_mask_area
            else:
                val = float(np.mean(diff))

            deltas.append(val)
            if val > max_delta:
                max_delta = val
                max_idx_in_chunk = i

        mean_diff = float(np.mean(deltas)) if deltas else 0.0
        is_active = mean_diff >= self.motion_threshold

        peak_global_idx = frame_indices[max_idx_in_chunk]
        peak_frame = frames[max_idx_in_chunk]

        return is_active, mean_diff, max_delta, peak_global_idx, peak_frame
