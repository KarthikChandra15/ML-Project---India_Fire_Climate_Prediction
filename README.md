# Indian Vegetation Fire Prediction

**72-Hour Forest & Vegetation Fire Risk Prediction Across India Using Machine Learning**  
*Course: **ES 335 — Machine Learning** | **Indian Institute of Technology (IIT) Gandhinagar***

- **Live Interactive Website (GitHub Pages):** [https://karthikchandra15.github.io/ML-Project---India_Fire_Climate_Prediction/](https://karthikchandra15.github.io/ML-Project---India_Fire_Climate_Prediction/)
- **Source Code Repository:** [https://github.com/KarthikChandra15/ML-Project---India_Fire_Climate_Prediction](https://github.com/KarthikChandra15/ML-Project---India_Fire_Climate_Prediction)

---

## 1. Problem Statement & Motivation

### What We Are Predicting
> **Given daily weather data, recent rainfall, land-cover fractions, and active fire history up to `23:59 IST` on day `D`, what is the probability that a `0.25° × 0.25°` (~27.8 km × 27.8 km) forest grid cell in India will have at least one forest fire during the next 72 hours (`[D+1, D+2, D+3]` IST)?**

### Why This Problem Matters
1. **Satellites are reactive, not predictive:** NASA satellites (MODIS and VIIRS) only detect fires *after* they start burning. Forest departments need a **3-day early warning** to position crews before a fire spreads.
2. **Satellites mix up crop burning and forest fires:** In India, about **20.5%** of all satellite fire detections are farmers burning leftover crop residue after harvest. If we train a model on raw satellite points without filtering, it learns farm burning schedules instead of forest fire risk.

---

## 2. Summary of Key Results (Executed in Google Colab Pro — T4 GPU)

| Stage / Metric | Result |
| :--- | :--- |
| **ERA5 Weather Data Processed** | **3,939,700 GRIB records** across **39 files** (`1947–2026`, covering India `6°N–38°N, 68°E–98°E`) |
| **NASA FIRMS Satellite Detections** | **2,806,767 raw points** → **2,216,077 unique fires** (`590,690` duplicate overlapping satellite passes removed) |
| **Forest vs. Crop Fire Separation** | **1,045,032 (`47.16%`) Forest Fires** · **453,496 (`20.46%`) Crop Fires** · **717,549 (`32.38%`) Uncertain/Mixed** |
| **Supervised ML Dataset Size** | **8,920,576 total forest cell-days** (`9,728` forest cells); **450,000 stratified benchmark rows** (`150K` Train, `150K` Val, `150K` Test) |
| **Best Overall Model (`ConvGRU`)** | **Test PR-AUC = `0.8028`** · **ROC-AUC = `0.9820`** · **F1 = `0.7328`** (**+81.8% higher PR-AUC** than Climatology baseline `0.4416`) |
| **Best Validation Model (`U-Net`)** | **Validation PR-AUC = `0.6944`** · **Test PR-AUC = `0.7945`** · **ROC-AUC = `0.9807`** · **F1 = `0.7253`** |
| **Best Tabular Model (`LightGBM`)** | **Test PR-AUC = `0.7930`** · **ROC-AUC = `0.9806`** · **Brier Score = `0.0310`** · **Calibration Error (ECE) = `0.0191`** |
| **Unseen Region Generalization (`LOEO`)** | **Mean PR-AUC = `0.6599`** · **Mean ROC-AUC = `0.9562`** across 8 held-out Indian ecoregions |
| **Automated Verification Checks** | **`7 / 7` Unit & Integration Tests PASSED (`0` failures)** |

---

## 3. Data Sources, Preprocessing & Anti-Leakage Steps

### 3.1 Data Sources
1. **ECMWF ERA5 Weather Reanalysis (`1947–2026`):** Hourly 0.25° gridded records of 2m temperature (`t2m`), 2m dewpoint (`d2m`), 10m wind components (`u10`, `v10`), total precipitation (`tp`), and surface solar radiation (`ssrd`).
2. **NASA FIRMS Active Fire Archive (`2000–2025`):** Active fire detections from **MODIS Collection 6.1** (1 km resolution) and **VIIRS S-NPP** (375 m resolution), combined with ESA WorldCover / MODIS IGBP forest and cropland fractions.

### 3.2 How We Processed the Data & Prevented Data Leakage
- **Time-Zone Alignment (`UTC+05:30` IST):** Shifted all hourly UTC weather records and satellite timestamps to **Indian Standard Time (IST)** before computing daily min/mean/max values, so afternoon fire weather matches afternoon satellite passes.
- **Duplicate Removal:** Found that `data_5.grib` and `data_6.grib` were 100% identical (`35,064` solar records) and removed `70,135` overlapping GRIB messages. Also removed `590,690` duplicate detections where two satellite passes saw the same fire within `0.01°` (~1.1 km) and 15 minutes.
- **5-Factor Forest vs. Crop Fire Filter:** Separated real forest fires (`Y = 1`) from agricultural crop burning (`Y = 0`) using static flare filtering, detection confidence (`≥ 0.30`), forest cover (`≥ 30%`), cropland cover (`≥ 50%`), and harvest burning windows (Oct–Nov and Mar–May).
- **Engineered Fire-Weather Features (strictly up to Day `D`):** Computed **Vapor Pressure Deficit (`VPD`, in kPa)** for air dryness, **Fosberg Fire Weather Index (`FFWI`)** for heat + wind + humidity danger, rolling `3/7/14/30-day` rainfall totals, consecutive dry days, and `1/3/7-day` recent fire counts.
- **Chronological Split with 3-Day Embargo Gaps (Buffer Days):**
  - **Train:** `2003–2011` (`150,000` rows)
  - **Validation:** `2012–2015` (`150,000` rows — used for hyperparameter tuning, isotonic probability calibration, and picking the alert threshold `θ = 0.324`)
  - **Holdout Test:** `2016–2025` (`150,000` rows — evaluated once at the end)
  - **3-Day Embargo Gap (Buffer):** Because our target looks 3 days ahead (`[D+1, D+2, D+3]`), we skip a **3-day buffer gap** at the boundary between Train, Validation, and Test sets so a 3-day prediction window at the end of Training never overlaps into Validation or Test.
- **No Fake Zero-Imputation:** The 1997–2008 10m wind gap in our ERA5 archive is kept as `NaN` rather than filling fake `0.0` wind speeds. All feature scalers are fit **only on the Training split**.

---

## 4. Models Compared (9 Models Across 3 Families)

| Model | Family | Validation PR-AUC | Holdout Test PR-AUC | Holdout Test ROC-AUC | Holdout Test F1 | Holdout Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `dl_convgru` | 3D Deep Learning | `0.6901` | **`0.8028`** | **`0.9820`** | **`0.7328`** | `0.0636` |
| `dl_spatiotemporal_unet` | 3D Deep Learning | **`0.6944`** | `0.7945` | `0.9807` | `0.7253` | `0.0697` |
| `dl_convlstm` | 3D Deep Learning | `0.6908` | `0.7940` | `0.9810` | `0.7257` | `0.0659` |
| `lightgbm` (Calibrated) | Tabular ML | `0.6679` | `0.7930` | `0.9806` | `0.7022` | **`0.0310`** |
| `xgboost` (Calibrated) | Tabular ML | `0.6609` | `0.7929` | `0.9808` | `0.7085` | `0.0314` |
| `random_forest` | Tabular ML | `0.6475` | `0.7866` | `0.9794` | `0.6983` | `0.0319` |
| `logistic_regression` | Tabular ML | `0.6361` | `0.7596` | `0.9721` | `0.6907` | `0.0324` |
| `persistence_7d_baseline` | Domain Baseline | `0.5512` | `0.6486` | `0.9316` | `0.6725` | `0.0499` |
| `climatology_baseline` | Domain Baseline | `0.3614` | `0.4416` | `0.8121` | `0.5101` | `0.0649` |

### Why We Use `PR-AUC`, `Brier Score`, and `ECE` Instead of Raw Accuracy
Only **7.47%** of forest grid-days on the holdout test set have a fire (`92.53%` are non-fire days). A dummy model that always predicts "no fire" gets **92.53% accuracy** while catching zero fires. Therefore, we evaluate models using:
- **`PR-AUC` (Precision–Recall Area Under Curve):** Measures how well the model catches rare fires without raising false alarms.
- **`Brier Score` & `ECE` (Expected Calibration Error):** Measure whether a predicted probability (e.g., `30%` risk) matches real-world fire frequency (`0.0` is perfect).

---

## 5. Code Location, Data Access & How to Rebuild

### 5.1 Repository Files
- [`index.html`](index.html) — Self-contained interactive ES 335 website (includes the Project Roadmap, ES 335 Rubric Navigator, 2 pipeline flowcharts, 10 `Chart.js` graphs, Interactive India Map with 12 pinned forest reserves, Alert Threshold Simulator, and Fire-Weather Physics Sandbox).
- [`India_Fire_Climate_Intelligence.ipynb`](India_Fire_Climate_Intelligence.ipynb) — Master Google Colab Pro notebook (42 cells, Sections `0–20`) containing all 19 modular Python scripts (`src/ifci/`) and pipeline execution logs.
- [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md) — Full 21-section written research report with all tables, error analyses, and ablation studies.
- [`README.md`](README.md) — Project overview, setup/rebuild guide, data access instructions, AI disclosure, and citations.

### 5.2 Data Access Instructions
1. **Pre-Computed Web Data (No 21 GB Download Needed to View Website):** All JSON and GeoJSON experimental outputs are embedded directly inside [`index.html`](index.html), so anyone can open the live GitHub Pages link (or double-click `index.html` locally) and use all charts, maps, and simulators immediately.
2. **Full Raw Dataset Access (For Re-Running in Google Colab Pro):**
   - **ERA5 GRIB Files (`1947–2026`):** Downloaded from the [Copernicus Climate Data Store (CDS) — ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/) and placed in Google Drive at `/content/drive/MyDrive/machine learning/`.
   - **NASA FIRMS API Key:** Obtain a free `FIRMS_MAP_KEY` from [https://firms.modaps.eosdis.nasa.gov/api/](https://firms.modaps.eosdis.nasa.gov/api/) and add it to **Google Colab Secrets (`🔑`)** under the name `FIRMS_MAP_KEY`. No API keys or tokens are ever hardcoded in this repository.

### 5.3 How to Run or Rebuild
- **To View the Website:** Open the live GitHub Pages URL ([https://karthikchandra15.github.io/ML-Project---India_Fire_Climate_Prediction/](https://karthikchandra15.github.io/ML-Project---India_Fire_Climate_Prediction/)) or double-click [`index.html`](index.html) locally.
- **To Re-Run the Full ML Pipeline:** Open [`India_Fire_Climate_Intelligence.ipynb`](India_Fire_Climate_Intelligence.ipynb) in Google Colab Pro (`T4` GPU, High-RAM), mount Google Drive, set `FIRMS_MAP_KEY` in Colab Secrets, and run Sections `0–20`. Section `20` runs the 7 automated verification checks (`UT-01` through `UT-06` and `IT-01`).

---

## 6. AI-Use Disclosure & Citations (ES 335 Requirement)

### 6.1 AI-Use Disclosure
In accordance with the **ES 335 (IIT Gandhinagar)** course requirements:
- **AI Tools Used:** Google Antigravity (Gemini-based coding assistant) was used to help structure the modular Python scripts (`src/ifci/`), optimize memory-safe streaming of the 39 ERA5 GRIB files in Google Colab, fix API pagination and timezone bugs, and build the HTML/Chart.js/Leaflet interactive website layout.
- **Student Work & Verification:** The problem formulation, chronological split design with 3-day embargo buffer gaps, physical fire-weather features (Tetens VPD, Fosberg FFWI), 5-factor forest vs. crop fire separation rules, full execution of all 20 stages on the real ERA5 and NASA FIRMS datasets in Google Colab Pro, and interpretation of model errors and limitations were done and verified by the authors (`7 / 7` automated verification checks passed).

### 6.2 External Data, Literature & Software Citations
1. **ERA5 Weather Reanalysis:** Hersbach, H., et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. Copernicus Climate Change Service (C3S).
2. **NASA FIRMS MODIS Active Fire Product (1 km):** Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The collection 6 MODIS active fire detection algorithm and fire products*. Remote Sensing of Environment, 178, 31–41.
3. **NASA FIRMS VIIRS Active Fire Product (375 m):** Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). *The New VIIRS 375 m active fire detection data product*. Remote Sensing of Environment, 143, 85–96.
4. **Fosberg Fire Weather Index (FFWI):** Fosberg, M. A. (1978). *Weather in wildland fire management: the fire weather index*. USDA Forest Service.
5. **Tetens Vapor Pressure Formula:** Tetens, O. (1930). *Über einige meteorologische Begriffe*. Zeitschrift für Geophysik, 6, 297–309.
6. **Land Cover & Forest Maps:** Zanaga, D., et al. (2022). *ESA WorldCover 10 m*; Friedl, M., & Sulla-Menashe, D. (2022). *MODIS Land Cover (MCD12C1)*; *Forest Survey of India (ISFR 2023)*.
7. **Software Libraries:** `PyTorch` (Paszke et al., 2019), `LightGBM` (Ke et al., 2017), `XGBoost` (Chen & Guestrin, 2016), `scikit-learn` (Pedregosa et al., 2011), `SHAP` (Lundberg & Lee, 2017), `xarray`, `cfgrib`, `Leaflet.js`, and `Chart.js`.
