# Color-X-ray-DRS  
**Deep Learning–Driven Chromatic X-Ray Imaging with Scintillator Film Stacks for Quantitative Densitometry**

---

## 1. What it does  
- Acquires RGB X-ray images through a stacked Cs/Cu-I + Mn + Bmpip₂SnBr₄ scintillator film  
- Segments materials by density with an Attention-U-Net  
- Regresses quantitative density (g cm⁻³) per pixel via an XGBoost model trained on RGB values  
- Outputs 2-D density maps and optional 3-D point clouds after multi-angle fusion  

---

## 2. Repo layout  
```
└─code
    ├─.ipynb_checkpoints
    ├─datasets
    │  ├─dataset_rgb
    │  │  └─newfoilder
    │  └─tryx
    │      ├─5
    │      ├─mask
    │      │  ├─test_attention
    │      │  └─train_attention
    │      ├─test
    │      │  ├─1st_manual
    │      │  └─images
    │      └─training
    │          ├─1st_manual
    │          └─images
    ├─nets
    │  └─__pycache__
    ├─pre
    └─__pycache__
```

---

## 3. One-line install  
```bash
git clone https://github.com/your-lab/Color-X-ray-DRS.git
cd Color-X-ray-DRS
pip install -r requirements.txt
```

---

## 4. Run inference (example)  
```bash
python src/infer.py --rgb data/sample.tif --out sample_density.tif --seg_model models/attn_unet.pth --reg_model models/xgb_density.json
```

---

## 5. Train your own  
Please view the detailed model training and deployment instructions in the 'README' in the 'code' folder

---

## 6. Performance (on test set)  
- Segmentation IoU: 0.81 (Attn-U-Net) vs 0.66 (baseline U-Net)  
- Density regression: R² = 0.889, RMSE = 0.622 g cm⁻³ (XGBoost)  

---

## 7. Citation  
If you use this code or the dataset, please cite:  
```
@article{your2024color,
  title={Color X-ray density regression with vapor-deposited multicolor scintillator films},
  author={H.Wang et al.h,
  journal={ Inf Funct Mater.},
  year={2025}
}
```

---

## 8. License  
MIT License - see `LICENSE` file.
