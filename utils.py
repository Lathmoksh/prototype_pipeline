import cv2
import numpy as np

def resize_and_pad(crop, target_size=224, background_color=(128, 128, 128)):
    """
    Resizes and pads an image crop to a target square size preserving aspect ratio.
    """
    h_c, w_c = crop.shape[:2]
    if h_c == 0 or w_c == 0:
        return np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
    
    # Scale factor to fit the crop inside target_size
    scale = target_size / max(h_c, w_c)
    new_w = int(w_c * scale)
    new_h = int(h_c * scale)
    
    # Resize keeping aspect ratio
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad evenly to target_size x target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=background_color)
    return padded

def apply_clahe(img, clahe=None):
    """
    Applies CLAHE histogram equalization on the BGR image.
    If a pre-constructed cv2.CLAHE object is provided, it is reused.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
