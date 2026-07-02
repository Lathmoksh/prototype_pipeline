import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils import resize_and_pad, apply_clahe
import argparse

LANE_THRESHOLD_COEFF = 0.22

def get_masked_patch(img_clahe, y_start, y_end, x_start, x_end):
    h, w = img_clahe.shape[:2]
    lane_w = x_end - x_start
    xc = (x_start + x_end) / 2.0
    
    x1 = int(xc - 0.7 * lane_w)
    x2 = int(xc + 0.7 * lane_w)
    
    x1_cl = max(0, x1)
    x2_cl = min(w, x2)
    
    crop = img_clahe[y_start:y_end, x1_cl:x2_cl]
    
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    lx1 = max(0, x_start - x1_cl)
    lx2 = min(crop.shape[1], x_end - x1_cl)
    mask[:, lx1:lx2] = 255
    
    bg = np.ones_like(crop) * 128
    masked_crop = np.where(mask[:, :, None] == 255, crop, bg)
    
    target_w = x2 - x1
    if masked_crop.shape[1] < target_w:
        pad_left = max(0, 0 - x1)
        pad_right = max(0, x2 - w)
        masked_crop = cv2.copyMakeBorder(
            masked_crop, 0, 0, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[128, 128, 128]
        )
        
    return masked_crop

def preprocess_patch(patch, transform, target_size=224):
    padded_img = resize_and_pad(patch, target_size)
    padded_img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(padded_img_rgb)
    return transform(pil_img).unsqueeze(0)

def adaptive_lane_features(img, x_start, x_end):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    # Trim by 4 pixels on each side, but ensure we don't trim more than half the width
    width = x_end - x_start
    trim = min(4, max(0, (width - 2) // 2))
    x_s = x_start + trim
    x_e = x_end - trim
    lane = gray[:, x_s:x_e]
    lane_w = x_e - x_s
    if lane_w <= 0:
        return 0.0, 0.0, 0.0

    flat = lane.flatten().astype(np.uint8).reshape(-1, 1)
    otsu_thresh, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = int(otsu_thresh)

    row_means_all = np.mean(lane, axis=1).astype(float)
    lane_mean_all = float(np.mean(row_means_all))
    row_cv = float(np.std(row_means_all)) / lane_mean_all if lane_mean_all > 0 else 0.0

    WINDOW_H, STRIDE = 224, 112
    min_dark = max(1, int(lane_w * 0.03))
    max_consec_ratio = 0.0
    max_diff_norm    = 0.0

    y = 0
    while y < h:
        y_end = min(y + WINDOW_H, h)
        win   = lane[y:y_end, :]
        win_h = y_end - y
        if win_h < 20:
            break
        dark_per_row = np.sum(win < otsu_thresh, axis=1)
        is_empty = dark_per_row < min_dark
        max_c = cur = 0
        for e in is_empty:
            if e: cur += 1; max_c = max(max_c, cur)
            else: cur = 0
        max_consec_ratio = max(max_consec_ratio, max_c / win_h)
        row_means = np.mean(win, axis=1).astype(float)
        row_diffs = np.abs(np.diff(row_means))
        if len(row_diffs) > 0 and lane_mean_all > 0:
            max_diff_norm = max(max_diff_norm, float(np.max(row_diffs)) / lane_mean_all)
        if y_end == h:
            break
        y += STRIDE

    return max_consec_ratio, row_cv, max_diff_norm

def detect_lanes(img, expected_lanes=10):
    orig_h, orig_w = img.shape[:2]
    
    # 1. Rotate to portrait if landscape
    if orig_w > orig_h:
        img_temp = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = orig_w, orig_h
        print("Rotated image to portrait orientation.")
    else:
        img_temp = img.copy()
        h, w = orig_h, orig_w
        
    # 2. Check if already a single lane
    min_dim = min(w, h)
    max_dim = max(w, h)
    aspect_ratio = min_dim / max_dim
    if min_dim < 350 or aspect_ratio <= 0.15:
        return [(0, w)], img_temp
        
    # 3. Scale to height 800 for normalized feature sizes
    scale = 800.0 / h
    small = cv2.resize(img_temp, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    col_means = np.mean(gray, axis=0)
    
    # 4. Thresholding
    min_val = np.min(col_means)
    max_val = np.max(col_means)
    thresh = min_val + (max_val - min_val) * 0.70
    is_lane = col_means < thresh
    
    # 5. Fill small holes inside lanes (background noise)
    in_hole = False
    start_idx = 0
    for idx, val in enumerate(is_lane):
        if not val and not in_hole:
            start_idx = idx
            in_hole = True
        elif val and in_hole:
            if idx - start_idx <= 2:
                is_lane[start_idx:idx] = True
            in_hole = False
    if in_hole and (len(is_lane) - start_idx <= 2):
        is_lane[start_idx:] = True
        
    # 6. Remove small objects outside lanes (bright noise)
    in_obj = False
    start_idx = 0
    for idx, val in enumerate(is_lane):
        if val and not in_obj:
            start_idx = idx
            in_obj = True
        elif not val and in_obj:
            if idx - start_idx < 5:
                is_lane[start_idx:idx] = False
            in_obj = False
    if in_obj and (len(is_lane) - start_idx < 5):
        is_lane[start_idx:] = False
        
    # 7. Extract intervals
    lanes = []
    in_lane = False
    start_idx = 0
    for idx, val in enumerate(is_lane):
        if val and not in_lane:
            start_idx = idx
            in_lane = True
        elif not val and in_lane:
            lanes.append((start_idx, idx))
            in_lane = False
    if in_lane:
        lanes.append((start_idx, len(is_lane)))
        
    # Filter by minimum width
    min_width = int(small.shape[1] * 0.02)
    valid_lanes = [l for l in lanes if (l[1] - l[0]) >= min_width]
    
    # Map back to original scale
    raw_lanes = []
    for start, end in valid_lanes:
        orig_x1 = int(start / scale)
        orig_x2 = int(end / scale)
        raw_lanes.append((orig_x1, orig_x2))
        
    # 8. Dynamic splitting logic to handle merged lanes
    if len(raw_lanes) == 0:
        return [(0, w)], img_temp
        
    # If raw detected count matches expected lanes, don't split!
    if len(raw_lanes) == expected_lanes:
        return raw_lanes, img_temp
        
    widths = [e - s for s, e in raw_lanes]
    median_w = np.median(widths)
    
    # Dynamically scale thresholds based on expected number of lanes
    max_allowed_ratio = 1.3 / expected_lanes
    ideal_lane_ratio = 0.95 / expected_lanes
    
    if median_w > max_allowed_ratio * w:
        median_w = ideal_lane_ratio * w
    
    final_lanes = []
    for s, e in raw_lanes:
        width = e - s
        ratio = width / median_w
        if ratio >= 1.6:
            num_merged = int(round(ratio))
            sub_w = width / num_merged
            for i in range(num_merged):
                sub_s = int(s + i * sub_w)
                sub_e = int(s + (i + 1) * sub_w)
                final_lanes.append((sub_s, sub_e))
        else:
            final_lanes.append((s, e))
            
    return final_lanes, img_temp

def merge_defects(detections):
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda d: d['y_range'][0])
    merged = []
    current = detections[0]
    
    for next_det in detections[1:]:
        curr_start, curr_end = current['y_range']
        next_start, next_end = next_det['y_range']
        
        if next_start <= curr_end and next_det['class'] == current['class']:
            current['y_range'] = (curr_start, max(curr_end, next_end))
            current['confidence'] = max(current['confidence'], next_det['confidence'])
        else:
            merged.append(current)
            current = next_det
    merged.append(current)
    return merged

def main():
    parser = argparse.ArgumentParser(description="Sheet Lane Segmentation & Classification")
    parser.add_argument("--lanes", type=int, default=10, help="Expected number of lanes (default: 10)")
    args = parser.parse_args()
    expected_lanes = args.lanes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Expected lanes target config: {expected_lanes}")

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8))

    model = models.efficientnet_b0()
    num_features = model.classifier[1].in_features
    class_names = ['accepted', 'gooping', 'skipping']
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    
    # Load newly trained dynamic weights
    weights_path = 'keshav.pth'
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_dir = "./input"
    output_dir = "./output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in '{input_dir}' folder. Please place some images to check there.")
        return

    print("\nRunning Sheet Lane Segmentation & Classification (Dynamic Model):")
    print("=" * 70)
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"\nProcessing {img_file}")
        lanes, img = detect_lanes(img, expected_lanes)
        h, w = img.shape[:2]
        print(f"  Dimensions (after rotation check): {w}x{h}px")
        img_clahe = apply_clahe(img, clahe)
        print(f"  Detected {len(lanes)} lane(s).")
        
        result_img = img.copy()
        lane_classes = []
        
        for lane_idx, (x_start, x_end) in enumerate(lanes):
            lane_w = x_end - x_start
            cv2.rectangle(result_img, (x_start, 0), (x_end, h), (0, 255, 0), 2)
            cv2.putText(result_img, f"Lane {lane_idx+1}", (x_start + 5, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            patch_h = 224
            stride = 224
            lane_detections = []
            cap_margin = int(h * 0.06)
            y_start = cap_margin
            y_limit = h - cap_margin
            max_goop = max_skip = 0.0
            
            while True:
                y_end = min(y_start + patch_h, y_limit)
                if y_end - y_start < patch_h and y_start > cap_margin:
                    y_start = max(cap_margin, y_limit - patch_h)
                    y_end = y_limit
                
                if y_end - y_start < 20:
                    break
                
                patch = get_masked_patch(img_clahe, y_start, y_end, x_start, x_end)
                
                try:
                    input_tensor = preprocess_patch(patch, transform, 224).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                        
                        prob_goop = probabilities[1] * 100
                        prob_skip = probabilities[2] * 100
                        
                        max_goop = max(max_goop, prob_goop)
                        max_skip = max(max_skip, prob_skip)
                        
                        GOOP_THRESHOLD = 5.0
                        SKIP_THRESHOLD = 55.0
                        
                        if prob_skip > SKIP_THRESHOLD:
                            patch_cls, conf = "skipping", prob_skip
                        elif prob_goop > GOOP_THRESHOLD:
                            patch_cls, conf = "gooping", prob_goop
                        else:
                            patch_cls = None
                        
                        if patch_cls:
                            lane_detections.append({
                                'y_range': (y_start, y_end),
                                'class': patch_cls,
                                'confidence': conf
                            })
                except Exception as e:
                    print(f"  [WARN] Inference skipped for Lane {lane_idx+1} at Y: {y_start}-{y_end}: {e}")
                
                if y_end == y_limit:
                    break
                y_start += stride
            
            # Model decision for the lane
            GOOP_THRESHOLD = 5.0
            SKIP_THRESHOLD = 55.0
            if max_skip > SKIP_THRESHOLD:
                model_cls = "skipping"
            elif max_goop > GOOP_THRESHOLD:
                model_cls = "gooping"
            else:
                model_cls = "accepted"

            # OpenCV analysis for the lane
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_s = min(x_start + 4, x_end)
            x_e = max(x_end - 4, x_start)
            lane_bright = float(np.mean(gray_img[:, x_s:x_e])) if x_e > x_s else 0.0

            OPENCV_BRIGHT_THRESHOLD = 130.0
            SKIP_CONSEC_RATIO = 0.02
            GOOP_ROW_CV = 0.08
            GOOP_DIFF_NORM = 0.25

            opencv_reliable = lane_bright >= OPENCV_BRIGHT_THRESHOLD
            cr, rcv, mdn = adaptive_lane_features(img, x_start, x_end)

            # 1. Suppress false gooping patch-level detections if OpenCV row-to-row variance is low globally
            SAFEGUARD_GOOP_CV = 0.09
            SAFEGUARD_GOOP_DIFF = 0.12
            if opencv_reliable and (rcv < SAFEGUARD_GOOP_CV and mdn < SAFEGUARD_GOOP_DIFF):
                lane_detections = [d for d in lane_detections if d['class'] != 'gooping']
                if model_cls == "gooping":
                    model_cls = "accepted"
            
            # Hybrid final classification
            final_cls = model_cls
            if model_cls == "skipping":
                if opencv_reliable and cr < SKIP_CONSEC_RATIO:
                    if max_skip >= 60.0:
                        final_cls = "gooping"
                        lane_detections = [{'y_range': (0, h), 'class': 'gooping', 'confidence': max_goop}]
                    else:
                        final_cls = "accepted"
                        lane_detections = []
            elif model_cls == "accepted":
                if opencv_reliable:
                    if cr >= SKIP_CONSEC_RATIO:
                        final_cls = "skipping"
                        lane_detections = [{'y_range': (0, h), 'class': 'skipping', 'confidence': cr * 100}]
                    elif rcv >= GOOP_ROW_CV and mdn >= GOOP_DIFF_NORM:
                        final_cls = "gooping"
                        lane_detections = [{'y_range': (0, h), 'class': 'gooping', 'confidence': max(rcv, mdn) * 100}]

            # Merge and draw defects
            merged_defects = merge_defects(lane_detections)
            for defect in merged_defects:
                y_s, y_e = defect['y_range']
                cls = defect['class']
                conf = defect['confidence']
                
                # Dynamic coloring: Red for gooping, Blue for skipping
                color = (0, 0, 255) if cls == 'gooping' else (255, 0, 0)
                cv2.rectangle(result_img, (x_start + 4, y_s), (x_end - 4, y_e), color, 3)
                
                label = f"{cls} {conf:.1f}%"
                cv2.putText(result_img, label, (x_start + 8, y_s + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"  -> Lane {lane_idx+1}: Found defect '{cls}' at Y: {y_s}-{y_e}px (Confidence: {conf:.1f}%)")
            
            lane_classes.append((lane_idx + 1, final_cls))
                
        base_name, ext = os.path.splitext(img_file)
        save_path = os.path.join(output_dir, f"{base_name}_result{ext}")
        cv2.imwrite(save_path, result_img)
        print(f"  Saved annotated result image to: {save_path}")
        
        print(f"\nLane classes for {img_file}:")
        for l_idx, l_cls in lane_classes:
            print(f"Lane {l_idx}: {l_cls}")
        print("-" * 50)

if __name__ == '__main__':
    main()
