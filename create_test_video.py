"""
Utility script to generate a synthetic border CCTV test video ('border_cctv_simulation.mp4').
Demonstrates all 8 core analytics capabilities of IBVAP:
  1. Human detection & multi-object tracking
  2. Vehicle detection & classification
  3. Automatic Number Plate Recognition (ANPR)
  4. Face detection
  5. Virtual fence tripwire intrusion detection
  6. Restricted zone breach detection
  7. Suspicious loitering activity
  8. Night-time / low-light lighting transition
"""

import os
import math
import cv2
import numpy as np

def create_border_simulation_video(output_path="border_cctv_simulation.mp4", num_seconds=12, fps=30):
    width, height = 1280, 720
    total_frames = num_seconds * fps

    # Check for bus.jpg to extract realistic vehicle and person sprites
    bus_img_path = "bus.jpg"
    if not os.path.exists(bus_img_path):
        from ultralytics.utils.downloads import download
        download("https://ultralytics.com/images/bus.jpg", bus_img_path)

    base_src = cv2.imread(bus_img_path)
    # Extract person sprite: [48, 398, 245, 902]
    person_crop = base_src[398:902, 48:245]
    # Extract vehicle sprite (bus): [22, 231, 804, 756]
    vehicle_crop = base_src[231:756, 22:804]

    # Stamp a clear high-contrast license plate on the vehicle crop bumper
    vh, vw = vehicle_crop.shape[:2]
    plate_w, plate_h = 160, 42
    px1 = int(vw * 0.45)
    py1 = int(vh * 0.78)
    # White reflective plate with black border
    cv2.rectangle(vehicle_crop, (px1, py1), (px1 + plate_w, py1 + plate_h), (245, 245, 245), -1)
    cv2.rectangle(vehicle_crop, (px1, py1), (px1 + plate_w, py1 + plate_h), (20, 20, 20), 2)
    # Clear Indian/border security format license plate text: DL01AB9876
    cv2.putText(vehicle_crop, "DL01AB9876", (px1 + 8, py1 + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (10, 10, 10), 2, cv2.LINE_AA)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[Sim] Generating {num_seconds}s ({total_frames} frames) border surveillance simulation at {width}x{height}...")

    # Road and outpost background
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky / distant hills
    bg[0:220, :] = (140, 150, 160)
    # Ground / terrain
    bg[220:, :] = (70, 95, 80)
    # Border security road
    road_pts = np.array([[150, height], [420, 220], [540, 220], [600, height]], dtype=np.int32)
    cv2.fillPoly(bg, [road_pts], (60, 60, 60))
    # Road lane markings
    cv2.line(bg, (375, height), (480, 220), (255, 255, 255), 2)

    # Border fence posts graphic on the right
    for fx in range(650, width, 55):
        cv2.line(bg, (fx, 250), (fx, 580), (45, 45, 45), 3)
    cv2.line(bg, (650, 310), (width, 310), (90, 90, 90), 2)
    cv2.line(bg, (650, 440), (width, 440), (90, 90, 90), 2)

    # Guard Outpost booth graphic on left
    cv2.rectangle(bg, (30, 260), (160, 450), (100, 110, 120), -1)
    cv2.rectangle(bg, (30, 260), (160, 450), (30, 30, 30), 2)
    cv2.rectangle(bg, (60, 290), (130, 360), (200, 220, 240), -1) # Window
    cv2.putText(bg, "BOP-ALPHA", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Initial positions
    # 1. Intruder 1: walks from top-center across the virtual border fence line (y=468, line is at y=0.65*720 = 468)
    p1_start_y = 350
    p1_end_y = 580
    p1_x = 720

    # 2. Loiterer 2: stands near outpost inside restricted zone
    p2_x = 880
    p2_y = 480

    # 3. Vehicle: drives down the road from y=240 to y=620
    v_start_y = 230
    v_end_y = 520

    for f_idx in range(total_frames):
        frame = bg.copy()
        alpha_progress = f_idx / float(total_frames)

        # -------------------------------------------------------------
        # Draw Intruder 1 (Walking downwards, crossing virtual fence)
        # -------------------------------------------------------------
        cur_p1_y = int(p1_start_y + (p1_end_y - p1_start_y) * (f_idx / (total_frames * 0.85)))
        cur_p1_y = min(p1_end_y, cur_p1_y)
        # Scale sprite based on perspective depth
        p1_scale = 0.35 + 0.30 * (cur_p1_y - p1_start_y) / (p1_end_y - p1_start_y)
        ph, pw = person_crop.shape[:2]
        nw, nh = int(pw * p1_scale), int(ph * p1_scale)
        p1_resized = cv2.resize(person_crop, (nw, nh))

        # Composite person 1 onto frame
        px1, py1 = p1_x, cur_p1_y - nh
        if py1 >= 0 and px1 + nw < width and py1 + nh < height:
            frame[py1:py1+nh, px1:px1+nw] = p1_resized

        # -------------------------------------------------------------
        # Draw Person 2 (Loitering inside restricted border zone)
        # -------------------------------------------------------------
        # Subtle loitering jitter
        jitter_x = int(math.sin(f_idx * 0.2) * 5)
        p2_scale = 0.50
        p2_w, p2_h = int(pw * p2_scale), int(ph * p2_scale)
        p2_resized = cv2.resize(person_crop, (p2_w, p2_h))
        px2, py2 = p2_x + jitter_x, p2_y - p2_h
        if py2 >= 0 and px2 + p2_w < width and py2 + p2_h < height:
            frame[py2:py2+p2_h, px2:px2+p2_w] = p2_resized

        # -------------------------------------------------------------
        # Draw Vehicle (Driving down border road with ANPR plate)
        # -------------------------------------------------------------
        if f_idx >= 30:  # Enters after frame 30
            v_progress = (f_idx - 30) / float(total_frames - 60)
            v_progress = min(1.0, max(0.0, v_progress))
            cur_vy = int(v_start_y + (v_end_y - v_start_y) * v_progress)
            cur_vx = int(320 + (220 - 320) * v_progress)
            v_scale = 0.28 + 0.40 * v_progress
            cur_vw, cur_vh = int(vw * v_scale), int(vh * v_scale)
            v_resized = cv2.resize(vehicle_crop, (cur_vw, cur_vh))

            vy1, vx1 = cur_vy, cur_vx
            if vy1 + cur_vh < height and vx1 + cur_vw < width and vx1 >= 0:
                frame[vy1:vy1+cur_vh, vx1:vx1+cur_vw] = v_resized

        # -------------------------------------------------------------
        # Night-Time Lighting Transition (Frames 210 to 360, 7s to 12s)
        # -------------------------------------------------------------
        if f_idx >= 210:
            dark_factor = max(0.28, 1.0 - 0.72 * ((f_idx - 210) / (total_frames - 210)))
            frame = (frame * dark_factor).astype(np.uint8)

        # Write frame
        out.write(frame)

    out.release()
    print(f"[Sim] Generated test surveillance video: {output_path}")

if __name__ == "__main__":
    create_border_simulation_video()
