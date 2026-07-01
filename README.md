# Winding Spool Defect Detection Pipeline (EfficientNet + OpenCV Hybrid)

A standalone, high-performance runtime pipeline designed to detect winding defect classes (`skipping` and `gooping`) on multi-spool winding machine sheets using a hybrid Neural Network (EfficientNet-B0) + Adaptive OpenCV feature verification engine.

---

## 📂 Folder Structure

```text
prototype_pipeline/
├── input/                  # Place raw input sheet images here (e.g., .jpg, .png)
├── output/                 # Labeled output images will be saved here
├── predict_sheet_dynamic.py # Main execution and decision engine
├── utils.py                # Visual processing and resizing utilities
├── keshav.pth              # PyTorch EfficientNet model weights file
├── requirements.txt        # Dependency packages configuration list
└── README.md               # Pipeline documentation
```

---

## ⚙️ Installation & Setup

Before running the pipeline on a new laptop, ensure Python (version `3.9` to `3.11` recommended) is installed.

1. **Clone or Copy the Repository**
   ```bash
   git clone https://github.com/Lathmoksh/prototype_pipeline.git
   cd prototype_pipeline
   ```

2. **Install Dependencies**
   Install PyTorch, OpenCV, NumPy, and Pillow automatically via pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Usage

Place your raw sheet photos in the `input/` folder and execute the script from your terminal:

### 1. Default (10-Lane Spool Winding Sheets)
If your winding sheet contains 10 spools, run:
```bash
python predict_sheet_dynamic.py --lanes 10
```
*(You can also simply run `python predict_sheet_dynamic.py` since the lane count defaults to 10).*

### 2. Custom Winding Layouts (e.g., 5-Lane Sheets)
If your sheet contains 5 spools, specify the target configuration explicitly:
```bash
python predict_sheet_dynamic.py --lanes 5
```

### 3. Single Pre-Cropped Spools
If you pass a single cropped spool image (e.g., width ~100px), the pipeline's **aspect ratio safeguard** automatically detects it and bypasses lane segmentation without needing any additional arguments:
```bash
python predict_sheet_dynamic.py
```

---

## 🛠️ How it Works (Technical Flow)

The pipeline integrates neural network inference with image processing safeguards in four logical steps:

1. **Adaptive Lane Segmentation**: The script scales the input image to a standard height of `800px` for resolution-independent boundary scanning. It rotates landscape images to portrait, filters out background noise, and dynamically splits merged drums using median widths or machine proportions.
2. **Deep Learning Inference**: Slices the lane vertically into `224x224px` patches and runs them through PyTorch (`EfficientNet-B0` loaded with `keshav.pth`).
3. **OpenCV Reassurance Safeguard**:
   * **Low-Exposure Guard**: Bypasses image processing if the lane brightness is `< 130`, directly trusting the model weights.
   * **Physical Spacing Verification (`consec_ratio`)**: Computes consecutive empty rows using sliding Otsu auto-threshold windows to verify the presence of winding gaps.
   * **Defect Spike Checks (`row_cv` & `max_diff_norm`)**: Computes intensity variance metrics to verify irregularities.
4. **Override Blending Decider**: Blends NN confidence with OpenCV metrics to neutralize false alarms (e.g., color-contrast shifts on clean spools) or correct classification errors (reverting false skipping predictions to gooping). Labeled defect rectangles are saved in the `output/` directory.
