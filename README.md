# Indian Vegetation Fire Prediction

**Machine Learning-Based 72-Hour Vegetation & Forest Fire Risk Prediction and Climate Intelligence for India**  
*Course: **ES 335 — Machine Learning** (Undergraduate Deliverable)*  
*Institution: **Indian Institute of Technology (IIT) Gandhinagar***

---

## 1. Project Summary & Core Research Question

**Research Question:**  
> *Given antecedent meteorological reanalysis up to day $D$ (in Indian Standard Time, $\text{UTC}+05:30$), cumulative multi-day drought memory, fractional land-cover composition, seasonal harmonic indicators, and trailing active-fire persistence, what is the calibrated probability $P(Y_{i,D}^{(72\text{h})} = 1 \mid \mathbf{X}_{i, \le D})$ that a $0.25^\circ \times 0.25^\circ$ ($\approx 27.8\text{ km} \times 27.8\text{ km}$) Indian forest grid cell $i$ will experience at least one satellite-confirmed forest fire during the forward 72-hour window $[D+1, D+2, D+3]$ IST?*

### Headline Experimental Findings (Executed in Google Colab Pro — T4 High-RAM)

| Metric / Stage | Executed Result |
| :--- | :--- |
| **Raw ERA5 Archive Audited** | **3,939,700 GRIB messages** across **39 files** (`1947–2026`, $0.25^\circ \times 0.25^\circ$ India domain $[6^\circ\text{N}–38^\circ\text{N}, 68^\circ\text{E}–98^\circ\text{E}]$) |
| **NASA FIRMS Detections Acquired** | **2,806,767 raw hotspots** $\to$ **2,216,077 deduplicated detections** (`1,682,607` VIIRS + `533,470` MODIS; `590,690` swath duplicates removed) |
| **3-Class Fire Attribution (`485,173` active grid-days)** | **`1,045,032` (`47.16%`) Forest Fires** · **`453,496` (`20.46%`) Agricultural Residue Fires** · **`717,549` (`32.38%`) Uncertain/Mosaic** |
| **Supervised Spatiotemporal Dataset** | **`8,920,576` total forest grid-day rows** (`1,348` days $\times$ `9,728` valid forest cells); **`450,000`-row stratified benchmark** (`150,000` per split) |
| **Holdout Test Champion (`dl_convgru`)** | **Test PR-AUC = `0.80280`** · **ROC-AUC = `0.98201`** · **F1 = `0.73284`** (**+81.8% PR-AUC** over Climatology baseline `0.44159`) |
| **Validation Champion (`dl_spatiotemporal_unet`)** | **Validation PR-AUC = `0.69442`** · **Test PR-AUC = `0.79450`** · **ROC-AUC = `0.98065`** · **F1 = `0.72529`** |
| **Best Calibrated Tabular Model (`lightgbm`)** | **Test PR-AUC = `0.79300`** · **ROC-AUC = `0.98059`** · **Brier = `0.03103`** · **ECE = `0.01908`** · **Pod = `63.79%`** at **FAR = `1.44%`** |
| **Spatial Generalization (LOEO across 8 Ecoregions)** | **Mean Unseen-Ecoregion PR-AUC = `0.6599`** · **Mean ROC-AUC = `0.9562`** (without lat/lon coordinates) |
| **Automated Verification Suite (`UT-01..06`, `IT-01`)** | **`7 / 7` Checks PASSED (`0` failures)** |

---

## 2. Live Interactive Website & Local Preview Instructions

The interactive ES 335 showcase website is located in [`website/index.html`](website/index.html) and comes pre-bundled with all real JSON/GeoJSON experimental outputs in [`website/data/`](website/data/).

### Features Included on the Interactive Website
- **ES 335 100-Point Rubric Navigator:** One-click jump cards mapped to all 10 grading criteria (80 pts Shared ML Reasoning + 20 pts Undergraduate Deliverable).
- **2 Architectural & Methodological SVG Flowcharts:**
  1. End-to-End 5-Stage Pipeline Architecture (ERA5 GRIB + NASA FIRMS $\to$ Attribution $\to$ Leak-Free 72h Engineering $\to$ 9 Models $\to$ Interactive Portal).
  2. 5-Factor Fire Attribution Decision Flow & 3-Day Embargo Temporal Split Diagram.
- **10 Interactive Chart.js Scientific Graphs:** Attribution Sensitivity, Ecoregion Fire Rates, 9-Model Benchmark PR-AUC, Precision–Recall Curves, Calibration Reliability Diagram, Deep Learning Training Curves, SHAP Feature Importance, Leave-One-Ecoregion-Out (LOEO) Spatial Generalization, and 1947–2020 Climate Baseline Trends.
- **3 Useful Interactive Scientific Tools (Criterion 9):**
  1. **Interactive $0.25^\circ$ India Geospatial Explorer & Cell Inspector (Leaflet):** Filter by model (`Calibrated LightGBM`, `Spatiotemporal U-Net`, `Hybrid Ensemble`), probability threshold, and ecoregion; click any cell to inspect its coordinates, ecoregion, $\text{VPD}_{\max}$, $\text{FFWI}_{\max}$, and calibrated 72h risk.
  2. **Interactive Operational Alert Threshold ($\theta$) & Confusion Matrix Simulator:** Drag the alert threshold $\theta \in [0.05, 0.90]$ to inspect real-time changes in `TP`, `FP`, `FN`, `TN`, `Precision`, `Recall (POD)`, `False Alarm Rate (FAR)`, and `F1` across the $N = 150,000$ holdout test grid-days.
  3. **Interactive Physical Fire-Weather "What-If" Sandbox:** Adjust $T_{\max}$, $\text{RH}_{\min}$, 10m Wind Speed, 7-day Rainfall, and 7-day Fire Persistence to compute live Tetens $\text{VPD}$, Fosberg $\text{FFWI}$, and 72-hour forest-fire probability.

### How to Open the Website Locally
Because the website loads JSON/GeoJSON files via `fetch('data/...')`, serve the `website/` folder over a local HTTP server (or simply run [`open_website.bat`](open_website.bat) on Windows):

```bash
# Option A: Python built-in HTTP server (recommended)
cd website
python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

```bash
# Option B: Node.js npx serve
npx serve website -l 8080
```

### How to Publish to GitHub Pages
1. Push this repository to GitHub (`main` branch).
2. In your GitHub repository, navigate to **Settings $\to$ Pages**, or run Cell 19 in [`India_Fire_Climate_Intelligence.ipynb`](India_Fire_Climate_Intelligence.ipynb) with `GITHUB_REPO_SLUG = "<username>/<repo>"` and `APPROVE_GITHUB_PAGES_DEPLOY = True` (using `GITHUB_TOKEN` in Google Colab Secrets), which automatically deploys `website/` to the `gh-pages` branch with `.nojekyll`.

---

## 3. Code Location & Repository Structure

| Path | Description |
| :--- | :--- |
| [`India_Fire_Climate_Intelligence.ipynb`](India_Fire_Climate_Intelligence.ipynb) | **Master Self-Contained Google Colab Pro Notebook** (42 cells, Sections 0–20). Automatically writes all 19 modular Python files into `src/ifci/` and executes the full pipeline with `PipelineState` checkpoints. |
| [`colab_project/modules/ifci/`](colab_project/modules/ifci/) | **Modular Python Source Package (`ifci` v3.0.0 — 19 modules):**<br>• `config.py`, `utils.py`, `runtime.py`, `state.py` — Configuration, IST utilities, hardware profiler, atomic state manager.<br>• `audit.py`, `era5.py`, `climate.py` — Pure-Python binary GRIB1/GRIB2 scanner, streaming IST daily Zarr builder, Tetens VPD & Fosberg FWI physics, 1947–2000 baseline.<br>• `firms.py`, `landcover.py`, `attribution.py` — NASA FIRMS API downloader, swath deduplication, ESA WorldCover/MODIS IGBP forest/crop masks, 5-factor attribution engine.<br>• `features.py`, `eda.py` — Leak-free 72h target engineering, 3-day temporal embargo splits, EDA tables.<br>• `models_ml.py`, `models_dl.py` — Climatology, 7d Persistence, Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch `ConvLSTM`, `ConvGRU`, `Spatiotemporal U-Net`.<br>• `evaluation.py`, `prediction.py` — Spatiotemporal block-bootstrap CIs, LOEO spatial holdout, SHAP, feature-family ablations, 72h map generator.<br>• `visualization.py`, `deploy.py`, `verification.py` — Publication figures, interactive website builder, GitHub Pages publisher, and 7-check automated test suite. |
| [`website/`](website/) | **Interactive ES 335 Research Website Bundle:**<br>• [`website/index.html`](website/index.html) — Self-contained interactive dashboard (flowcharts, 10 Chart.js graphs, 3 interactive simulators, ES 335 rubric map).<br>• [`website/data/project_summary.json`](website/data/project_summary.json) — Complete experimental metrics, PR/calibration curves, SHAP, LOEO, and climate trends.<br>• [`website/data/latest_72h_prediction.json`](website/data/latest_72h_prediction.json) — 72-hour forest-fire probability forecasts across 3,500 Indian forest grid cells.<br>• [`website/data/active_fire_monitor.geojson`](website/data/active_fire_monitor.geojson) — Attributed active-fire GeoJSON detections (`forest_fire`, `agricultural_fire`, `other_or_uncertain`). |
| [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md) | **Complete 21-Section Final Research Report** with all tables, physical equations, error analyses, and ablation results. |
| [`SCIENTIFIC_METHODOLOGY.md`](SCIENTIFIC_METHODOLOGY.md) | Formal mathematical specification of Tetens VPD, Fosberg FWI, 5-factor fire attribution, leak-free 72h target alignment, and block-bootstrap inference. |
| [`CODE_AUDIT_V2.md`](CODE_AUDIT_V2.md) & [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) | Binary GRIB1 header audit across all 39 ERA5 files (`3,939,700` messages) and verification suite logs. |
| [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) | Step-by-step instructions for running or resuming any stage in Google Colab Pro. |

---

## 4. Data Access, Setup & Rebuild Instructions

### 4.1 Data Sources & Access Instructions
1. **ECMWF ERA5 Hourly Reanalysis (`1947–2026`, $0.25^\circ \times 0.25^\circ$ India Domain):**
   - **Source:** Copernicus Climate Data Store (CDS) — *ERA5 hourly data on single levels from 1940 to present* (`reanalysis-era5-single-levels`, Hersbach et al., 2020).
   - **Variables Used:** `2m_temperature` (`t2m`), `2m_dewpoint_temperature` (`d2m`), `10m_u_component_of_wind` (`u10`), `10m_v_component_of_wind` (`v10`), `total_precipitation` (`tp`), and `surface_solar_radiation_downwards` (`ssrd`).
   - **Google Drive Path:** Place the 39 raw `.grib` archive files (`data_0.grib` through `data_16.grib`, `1991-1993.grib` through `2025-2026.grib`) inside `/content/drive/MyDrive/machine learning/`.
   - **Pre-Computed Website Outputs:** All aggregated JSON/GeoJSON outputs required to view and interact with the website are already committed inside [`website/data/`](website/data/) (`< 2 MB` total), so **graders do not need to download the 21 GB raw GRIB archive to inspect or rebuild the interactive website**.

2. **NASA FIRMS Active-Fire Archive (`2000–2025`):**
   - **Source:** NASA Fire Information for Resource Management System (FIRMS) API (`/api/area/csv/{FIRMS_MAP_KEY}/{SOURCE}/68,6,98,38/{DAY_RANGE}/{DATE}`) covering `MODIS_SP` (Collection 6.1, 1 km) and `VIIRS_SNPP_SP` (375 m).
   - **Credential Handling:** Obtain a free API Map Key at [https://firms.modaps.eosdis.nasa.gov/api/](https://firms.modaps.eosdis.nasa.gov/api/) and store it in **Google Colab Secrets (`🔑`)** under the key name `FIRMS_MAP_KEY`. No API keys or tokens are ever hardcoded in source code or committed to the repository (`zero_credentials_or_tokens_in_bundle = true`).

### 4.2 Rebuilding the Website Locally (Without Re-Running Multi-Hour Colab Training)
To rebuild [`website/index.html`](website/index.html) from [`website/data/project_summary.json`](website/data/project_summary.json):
```bash
node colab_project/build_es335_showcase_website.js
```

### 4.3 Full End-to-End Rebuild in Google Colab Pro
1. Open [`India_Fire_Climate_Intelligence.ipynb`](India_Fire_Climate_Intelligence.ipynb) in **Google Colab Pro** (`T4` or `L4` GPU, High-RAM enabled).
2. Mount Google Drive at `/content/drive` and ensure `FIRMS_MAP_KEY` is added in Colab Secrets (`🔑`).
3. Run **Sections 0–20** sequentially. Every stage writes atomic checkpoints to `/content/drive/MyDrive/India_Fire_Climate_Intelligence/state/pipeline_state.json`, so completed stages automatically resume if a Colab session disconnects.
4. Run **Section 20 (`verification.run_verification_suite()`)** to verify all 7 unit and integration checks (`UT-01` through `UT-06` and `IT-01`).

---

## 5. AI-Use Disclosure & External Citations (ES 335 Requirement)

### 5.1 AI-Use Disclosure
In accordance with the **ES 335 (IIT Gandhinagar)** course policy on transparency and academic integrity, we disclose the following use of AI assistance during this project:
- **AI Tools Used:** Google Antigravity (Gemini-powered agentic coding assistant) was used to assist with:
  1. Refactoring the 19 modular Python scripts in `src/ifci/` for memory-safe streaming of the 39 ERA5 GRIB files (`3,939,700` messages) within Google Colab's disk/RAM constraints.
  2. Debugging API pagination (`5-day` chunking for NASA FIRMS `/api/area/csv/`) and timezone alignment (`UTC+05:30` IST naive vs. aware timestamps).
  3. Structuring the HTML/CSS/SVG/Chart.js/Leaflet interactive research portal (`website/index.html`) and formatting documentation tables.
- **Student Ownership & Verification:** All research questions, temporal split choices (`2003–2011` Train, `2012–2015` Validation, `2016–2025` Holdout Test with 3-day embargo windows), physical fire-weather formulations (Tetens VPD, Fosberg FWI), 5-factor forest vs. agricultural fire attribution criteria, execution of all 20 pipeline stages on real ERA5 and NASA FIRMS data in Google Colab Pro, and interpretation of errors and limitations were verified and validated by the project authors. Every numeric metric reported on the website and in the report is produced directly by the executed Colab pipeline and verified by the automated test suite (`7/7 PASSED`).

### 5.2 External Data, Scientific Literature & Software Citations
1. **ERA5 Reanalysis:** Hersbach, H., et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
2. **NASA FIRMS MODIS Active Fire Product:** Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The collection 6 MODIS active fire detection algorithm and fire products*. Remote Sensing of Environment, 178, 31–41.
3. **NASA FIRMS VIIRS 375 m Active Fire Product:** Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). *The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment*. Remote Sensing of Environment, 143, 85–96.
4. **Fosberg Fire Weather Index (FFWI):** Fosberg, M. A. (1978). *Weather in wildland fire management: the fire weather index*. In Proceedings of the Conference on Sierra Nevada Meteorology (pp. 1–4). USDA Forest Service.
5. **Tetens Saturation Vapor Pressure Equation:** Tetens, O. (1930). *Über einige meteorologische Begriffe*. Zeitschrift für Geophysik, 6, 297–309.
6. **ESA WorldCover / MODIS Land Cover:** Zanaga, D., et al. (2022). *ESA WorldCover 10 m 2021 v200*; Friedl, M., & Sulla-Menashe, D. (2022). *MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 0.05Deg CMG (MCD12C1)*.
7. **Forest Survey of India (FSI):** *India State of Forest Report (ISFR) 2021 & 2023*, Ministry of Environment, Forest and Climate Change, Government of India.
8. **Open-Source Software Libraries:** `PyTorch` (Paszke et al., 2019), `LightGBM` (Ke et al., 2017), `XGBoost` (Chen & Guestrin, 2016), `scikit-learn` (Pedregosa et al., 2011), `SHAP` (Lundberg & Lee, 2017), `xarray` (Hoyer & Hamman, 2017), `cfgrib` (ECMWF), `Leaflet.js`, and `Chart.js`.
