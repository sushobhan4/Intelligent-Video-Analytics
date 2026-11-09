import re
import cv2
import easyocr
from datetime import datetime, timedelta

import torch

class OSDTimestampExtractor:
    def __init__(self, crop_box=(40, 115, 60, 880)):
        self.crop_box = crop_box
        self.reader = None
        self.base_datetime = None
        self.base_frame = 0
        self.last_parsed_str = None
        self.last_ocr_frame = -999

    def _init_reader(self):
        if self.reader is None:
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def extract(self, frame, frame_idx, fps, force_ocr=False):
        # If we already have a valid base_datetime and this call is close to last OCR,
        # we can compute exact second offset without incurring OCR latency
        if not force_ocr and self.base_datetime is not None and (frame_idx - self.last_ocr_frame < int(fps * 2)):
            delta_sec = (frame_idx - self.base_frame) / max(fps, 1.0)
            cur_dt = self.base_datetime + timedelta(seconds=delta_sec)
            return cur_dt.strftime("%d-%m-%Y %H:%M:%S")

        y1, y2, x1, x2 = self.crop_box
        h, w = frame.shape[:2]
        if h != 1440 or w != 2560:
            y1 = int(h * 0.027)
            y2 = int(h * 0.080)
            x1 = int(w * 0.023)
            x2 = int(w * 0.35)

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

        self._init_reader()
        res = self.reader.readtext(thresh)
        raw_text = ' '.join([t[1] for t in res])

        # Normalize common OCR noise
        raw_text = raw_text.replace('+', ' ').replace(';', ':').replace('c', ':')
        match = re.search(r"(\d{2})[-/\s]+(\d{2})[-/\s]+(\d{4}).*?(\d{2})[:\s]+(\d{2})[:\s]+(\d{2})", raw_text)
        if match:
            day, mon, year, hour, minute, sec = map(int, match.groups())
            try:
                dt = datetime(year, mon, day, hour, minute, sec)
                self.base_datetime = dt
                self.base_frame = frame_idx
                self.last_ocr_frame = frame_idx
                self.last_parsed_str = dt.strftime("%d-%m-%Y %H:%M:%S")
                return self.last_parsed_str
            except ValueError:
                pass

        if self.base_datetime is not None:
            delta_sec = (frame_idx - self.base_frame) / max(fps, 1.0)
            cur_dt = self.base_datetime + timedelta(seconds=delta_sec)
            return cur_dt.strftime("%d-%m-%Y %H:%M:%S")

        return f"Frame {frame_idx}"


if __name__ == "__main__":
    extractor = OSDTimestampExtractor()
    for f_idx in [0, 13000, 47000]:
        img_name = f"frame_{f_idx}.jpg" if f_idx > 0 else "test_frame0.jpg"
        img = cv2.imread(img_name)
        ts = extractor.extract(img, f_idx, 33.3, force_ocr=True)
        print(f"Extracted timestamp for frame {f_idx}: {ts}")
