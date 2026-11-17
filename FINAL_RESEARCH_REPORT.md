# India Fire & Climate Intelligence: Machine Learning-Based Forest Fire Risk Prediction and Climate Intelligence for India

**Academic Context:** Final-Year B.Tech Machine Learning Research Project, Indian Institute of Technology (IIT) Gandhinagar  
**Execution Environment:** 100% Google Colab Pro (`T4 High-RAM`, 51.0 GB RAM, 15.0 GB VRAM) + Google Drive Persistent Storage  
**Report Generated (UTC):** `2026-11-17T13:57:11Z`  
**Verification Suite Status:** `7 / 7 PASSED` (`UT-01`..`UT-06` + `IT-01` Real ERA5 Zarr vs. Raw GRIB Decode)

---

## 1. Abstract

This study develops, validates, and deploys a leak-free machine-learning and spatiotemporal deep-learning framework to predict the probability of a **forest-associated active-fire detection** inside a $0.25^\circ \times 0.25^\circ$ grid cell across India over the **next 72 hours (`[D+1, D+2, D+3]` IST)** using strictly antecedent meteorological, drought-memory, land-cover, and fire-persistence features ($\le D$). Auditing **39 raw ERA5 GRIB1 archives (`3,939,700` messages, `1947–2026`)** and **2,806,767 NASA FIRMS active-fire detections (`2,216,077` deduplicated detections across MODIS Collection 6.1 and VIIRS 375m)**, we separate forest-associated fires (`47.16%`) from agricultural crop-residue burning (`20.46%`) and uncertain/industrial thermal anomalies (`32.38%`). On the strictly independent **2016–2025 holdout test split**, spatiotemporal deep learning (**`ConvGRU`**: `PR-AUC = 0.80280`, `ROC-AUC = 0.98201`, `F1 = 0.73284`; **`Spatiotemporal U-Net`**: `Val PR-AUC = 0.69442`, `Test PR-AUC = 0.79450`) and validation-calibrated gradient-boosted trees (**`LightGBM`**: `Test PR-AUC = 0.79300`, `Brier = 0.03103`, `ECE = 0.01908`; **`XGBoost`**: `Test PR-AUC = 0.79286`, `F1 = 0.70851`) achieve an **+81.8% relative improvement in PR-AUC** over Seasonal Climatology (`0.44159`) and a **+23.8% relative gain** over 7-Day Persistence (`0.64861`).

---

## 2. Introduction

Wildland fires across India's forest ecoregions—particularly the dry deciduous forests of the Central Indian Highlands, the subtropical hills of Northeast India, the Western and Eastern Ghats, and the Western Himalayas—threaten biodiversity, forest carbon stocks, and regional air quality. However, satellite active-fire observations over India are heavily confounded by seasonal agricultural crop-residue burning (post-monsoon Kharif paddy stubble in October–November and pre-monsoon Rabi wheat straw in March–May) and static industrial heat sources. A scientifically defensible forest-fire risk forecasting system must therefore:
1. Convert hourly reanalysis and satellite overpass timestamps to **Indian Standard Time (`Asia/Kolkata`, `UTC+05:30`)**,
2. De-accumulate ERA5 precipitation and solar radiation fluxes without silent zero-imputation of missing variables,
3. Scientifically distinguish forest-associated active-fire detections from agricultural and uncertain anomalies, and
4. Enforce strict chronological splits (`1947–2000` climate baseline, `2001–2010` train, `2011–2015` validation/calibration, `2016–2025` holdout test) with a 3-day boundary embargo.

---

## 3. Research Problem & Questions Addressed

**Primary Research Question:**
> *"How accurately can machine learning models predict the occurrence of forest-associated active-fire detections across India using antecedent meteorological and environmental conditions?"*

**Empirical Findings for Sub-Questions 1–7:**
1. **Top Meteorological Drivers:** Peak afternoon Vapour Pressure Deficit (`vpd_max_kpa`, $r = +0.2874$), Daily Maximum Temperature (`t2m_max_c`, $r = +0.1880$), Afternoon Minimum Relative Humidity (`rh_min_pct`, $r = -0.1203$), and 7-day cumulative precipitation (`tp_7d_mm`, $r = -0.0575$).
2. **Feature-Group Contributions:** Dropping antecedent fire lags reduces Test PR-AUC by $-0.09370$ (`0.80202` $\to$ `0.70832`); dropping spatial/seasonal context reduces Test PR-AUC by $-0.05111$ (`0.80202` $\to$ `0.75091`); dropping thermodynamic/VPD variables reduces Test PR-AUC by $-0.00450$.
3. **Pure Weather Predictability:** Even with all antecedent fire features removed (`without_antecedent_fire`), meteorological + land-cover + seasonal features achieve **`Test PR-AUC = 0.70832`** and **`Test ROC-AUC = 0.97329`** (vs. `0.44159` for Climatology).
4. **Deep Learning vs. Classical ML:** Spatiotemporal 2D neural architectures (`ConvGRU`, `Spatiotemporal U-Net`, `ConvLSTM`) outperform classical models in ranking/discrimination (`Test PR-AUC = 0.80280`, `F1 = 0.73284` vs. `0.79300` / `0.70851`), while isotonically calibrated `LightGBM`/`XGBoost` achieve superior probability calibration (`Brier = 0.03103`, `ECE = 0.01908`).
5. **Spatial Generalisation (`LOEO`):** Under Leave-One-Ecoregion-Out spatial holdout (with `latitude`/`longitude` removed), physical models transfer strongly to unseen major forest belts: `Central Indian Highlands` (`LOEO PR-AUC = 0.78910`), `Northeast Hills` (`0.73087`), `Western Ghats` (`0.73062`), and `Eastern Ghats` (`0.72968`).
6. **Probability Calibration:** Validation-only Isotonic Regression reduces holdout Expected Calibration Error (`ECE`) to `0.01348` (`Logistic Regression`) and `0.01908` (`LightGBM`) and cuts Brier Score by $>52\%$ relative to Climatology (`0.06489` $\to$ `0.03103`).
7. **Attribution & Sensor Sensitivity:** Across `strict_forest` ($\ge 0.40$), `baseline_forest` ($\ge 0.30$), and `inclusive_forest` ($\ge 0.20$) regimes, the forest-associated share varies smoothly (`45.27%` $\to$ `47.16%` $\to$ `55.31%`).

---

## 4. Objectives

1. Audit all 39 ERA5 GRIB1 files on Google Drive without modifying raw data.
2. Process hourly ERA5 messages into IST daily Zarr stores with verified numerical equivalence (`max_abs_diff = 0.0`) against raw GRIB decoding.
3. Acquire, validate, and deduplicate multi-year NASA FIRMS `MODIS_SP` and `VIIRS_SNPP_SP` observations over India.
4. Attribute detections into `forest_fire`, `agricultural_fire`, and `other_or_uncertain`.
5. Train and compare physical baselines, classical ML models, and 2D spatiotemporal deep neural networks on leak-free 72-hour forward targets.
6. Export a validated 9-section interactive research website for GitHub Pages.

---

## 5. Literature Context and Related Approaches

- **ERA5 Reanalysis:** Hersbach et al. (2020), *The ERA5 global reanalysis*, Q. J. R. Meteorol. Soc., 146(730), 1999–2049.
- **MODIS Active Fire Product:** Giglio et al. (2016), *The Collection 6 MODIS active fire detection algorithm and fire products*, Remote Sensing of Environment, 178, 31–41.
- **VIIRS 375m Active Fire Product:** Schroeder et al. (2014), *The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment*, Remote Sensing of Environment, 143, 85–96.
- **Fire-Weather Indices:** Fosberg (1978), *Weather in wildland fire management: the Fire Weather Index*; Keetch & Byram (1968), *A Drought Index for Forest Fire Control*, USDA Forest Service Research Paper SE-38.
- **Indian Forest Biogeography:** Champion & Seth (1968), *A Revised Survey of the Forest Types of India*; Forest Survey of India (FSI), *India State of Forest Report (ISFR)*.
- **Spatiotemporal Deep Learning:** Shi et al. (2015), *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*, NeurIPS; Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI.

---

## 6. Data Sources & Verified Archive Coverage

### 6.1 Verified Coverage Across Intended Chronological Research Periods
| Period | Intended Years | Total Years | Years With Any Var | Years With All 6 Core Vars | Status & Scientific Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `climate_baseline` | `1947–2000` | 54 | 54 | 38 | `PARTIAL` (`tp` starts `1948-01-01`; `d2m` gap `1948–1960`; `u10/v10` gap `1997–2000`). Used strictly for climate baseline. |
| `train` | `2001–2010` | 10 | 10 | 2 | `PARTIAL` (`t2m, d2m, tp, ssrd` 100% complete; `u10/v10` gap `2001–2008`, complete in `2009–2010`). |
| `validation` | `2011–2015` | 5 | 5 | 5 | **`COMPLETE`** (all 6 core variables 100% present). |
| `test_holdout` | `2016–2025` | 10 | 10 | 10 | **`COMPLETE`** (all 6 core variables 100% present). |

### 6.2 Per-Variable ERA5 Completeness Across 3,939,700 Audited GRIB Messages
| Variable ID | Canonical Var | Step Type | Start (UTC) | End (UTC) | Expected Steps | Present Steps | Missing Steps | Completeness (%) | Gaps | Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `t2m@surface:0` | `t2m` | `instant` | `1947-01-01` | `2025-12-31 23:00` | 692,520 | 692,520 | 0 | **100.000%** | 0 | 8 |
| `ssrd@surface:0` | `ssrd` | `accum` | `1947-01-01` | `2026-10-24 20:00` | 699,645 | 699,645 | 0 | **100.000%** | 0 | 9 |
| `tp@surface:0` | `tp` | `accum` | `1948-01-01` | `2025-12-31 23:00` | 683,760 | 683,760 | 0 | **100.000%** | 0 | 6 |
| `d2m@surface:0` | `d2m` | `instant` | `1947-01-01` | `2025-12-31 23:00` | 692,520 | 578,544 | 113,976 | **83.542%** | 1 (`1948–1960`) | 6 |
| `u10@surface:0` | `u10` | `instant` | `1947-01-01` | `2026-10-22 11:00` | 699,588 | 594,396 | 105,192 | **84.964%** | 1 (`1997–2008`) | 12 |
| `v10@surface:0` | `v10` | `instant` | `1947-01-01` | `2026-10-22 11:00` | 699,588 | 594,396 | 105,192 | **84.964%** | 1 (`1997–2008`) | 12 |
| `e@surface:0` | `e` | `accum` | `2023-01-01` | `2025-12-31 23:00` | 26,304 | 26,304 | 0 | **100.000%** | 0 | 1 |

---

## 7. Data Preprocessing & Manual GRIB Verification (`IT-01`)

All GRIB1 messages were deduplicated (`data_5.grib` == `data_6.grib`: `35,064` identical fields; `data_15.grib` / `data_20.grib`: `26,304` shared fields; `data.grib` / `data_21.grib`: `8,760` shared fields), aligned onto the canonical $0.25^\circ$ India grid (`125 x 119`), and aggregated into 24-hour **IST (`UTC+05:30`)** daily Zarr stores. Manual single-day verification (`1947-01-02 IST`, `120` raw GRIB messages) confirmed **`max_abs_diff = 0.0` across all 15 daily variables**.

---

## 8. Fire Attribution Methodology & Sensitivity Analysis

Across **2,806,767 raw FIRMS detections** over India (`2,216,077` after $0.01^\circ$/15-min cross-swath deduplication; `1,682,607` VIIRS + `533,470` MODIS across `485,173` active grid-days):

### 8.1 Attribution Sensitivity Across Threshold Regimes
| Regime | Forest Frac Thresh | Crop Frac Thresh | Min Conf Score | Forest-Associated (`forest_fire`) | Forest (%) | Agricultural (`agricultural_fire`) | Agri (%) | Other / Uncertain (`other_or_uncertain`) | Uncertain (%) | Forest FRP (MW) | Agri FRP (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `strict_forest` | `0.40` | `0.55` | `0.50` | `1,003,176` | **45.27%** | `437,665` | 19.75% | `775,236` | 34.98% | `19,947,276.0` | `3,976,343.5` |
| `baseline_forest` | `0.30` | `0.45` | `0.30` | `1,045,032` | **47.16%** | `453,496` | 20.46% | `717,549` | 32.38% | `20,589,960.0` | `4,163,069.8` |
| `inclusive_forest` | `0.20` | `0.40` | `0.20` | `1,225,654` | **55.31%** | `507,809` | 22.91% | `482,614` | 21.78% | `21,708,954.0` | `4,480,307.0` |

### 8.2 72-Hour Fire Incidence by Indian Ecoregion (`8,920,576` Supervised Forest Grid-Days)
| Ecoregion | Total Grid-Days | 72-Hour Forest-Fire Rate | 72-Hour Agricultural-Fire Rate |
| :--- | :---: | :---: | :---: |
| `central_indian_highlands_forest` | `834,470` | **17.648%** | `0.000%` |
| `northeast_hills_forest` | `897,743` | **16.521%** | `0.000%` |
| `western_ghats_forest` | `516,271` | **6.342%** | `0.000%` |
| `eastern_ghats_forest` | `533,694` | **5.864%** | `0.000%` |
| `western_himalayas_forest` | `727,181` | **2.168%** | `0.000%` |
| `eastern_himalayas_forest` | `161,392` | **1.517%** | `0.000%` |
| `deccan_peninsular_mosaic` | `4,937,128` | `0.166%` | **2.637%** |
| `andaman_nicobar_evergreen_forest` | `312,697` | `0.118%` | `0.000%` |

---

## 9. Meteorological & Antecedent Feature Engineering

Linear correlation analysis across the `8,920,576` forest grid-days confirmed physically consistent relationships with `target_forest_fire_next72h`:
- `vpd_max_kpa`: **$r = +0.2874$**
- `t2m_max_c`: **$r = +0.1880$**
- `rh_min_pct`: **$r = -0.1203$**
- `tp_7d_mm`: **$r = -0.0575$**

---

## 10. Target Construction (`target_forest_fire_next72h`)

Constructed over `1,348` daily slices across `9,728` valid forest/woodland cells (`8,920,576` full rows: `3,385,344` Train [`pos_rate = 2.332%`], `2,821,120` Validation [`pos_rate = 3.720%`], and `2,714,112` Holdout Test [`pos_rate = 7.461%`]), with a stratified `450,000`-row benchmark (`150,000` per split) for classical tabular models and full `2,967,040` / `2,937,856` spatial pixel evaluations for 2D spatiotemporal deep learning.

---

## 11–14. Experimental Results: Classical ML vs. Spatiotemporal Deep Learning

| Rank (Val PR-AUC) | Model | Family | Val Thresh | Val PR-AUC (`2011–2015`) | Val ROC-AUC | Holdout Test PR-AUC (`2016–2025`) | Holdout ROC-AUC | Holdout F1 | Holdout Precision | Holdout Recall | Holdout Brier $\downarrow$ | Holdout Log-Loss $\downarrow$ | Holdout ECE $\downarrow$ |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`dl_spatiotemporal_unet`** | Spatiotemporal DL | `0.7439` | **`0.69442`** | `0.97939` | `0.79450` | `0.98065` | `0.72529` | `0.71892` | `0.73178` | `0.06968` | `0.24273` | `0.15007` |
| **2** | **`dl_convlstm`** | Spatiotemporal DL | `0.7435` | `0.69078` | `0.98015` | `0.79396` | `0.98097` | `0.72573` | `0.71866` | `0.73295` | `0.06593` | `0.21232` | `0.12151` |
| **3** | **`dl_convgru`** | Spatiotemporal DL | `0.7394` | `0.69008` | **`0.98085`** | **`0.80280`** | **`0.98201`** | **`0.73284`** | `0.71511` | **`0.75147`** | `0.06356` | `0.21071` | `0.12276` |
| **4** | **`lightgbm` (Cal.)** | Classical ML | `0.3241` | `0.66788` | `0.97697` | `0.79300` | `0.98059` | `0.70221` | **`0.78090`** | `0.63792` | **`0.03103`** | **`0.09881`** | `0.01908` |
| **5** | **`xgboost` (Cal.)** | Classical ML | `0.3115` | `0.66090` | `0.97666` | `0.79286` | `0.98083` | `0.70851` | `0.77145` | `0.65506` | `0.03138` | `0.09944` | `0.02054` |
| **6** | **`random_forest` (Cal.)** | Classical ML | `0.3103` | `0.64748` | `0.97519` | `0.78661` | `0.97943` | `0.69831` | `0.76886` | `0.63962` | `0.03194` | `0.10175` | `0.02153` |
| **7** | **`logistic_regression` (Cal.)** | Classical ML | `0.3441` | `0.63610` | `0.96901` | `0.75956` | `0.97212` | `0.69071` | `0.74048` | `0.64721` | `0.03238` | `0.10855` | **`0.01348`** |
| **8** | **`persistence_7d_baseline`** | Physical Baseline | `0.2173` | `0.54992` | `0.92825` | `0.64861` | `0.93161` | `0.67248` | `0.62643` | `0.72585` | `0.04990` | `0.16543` | `0.04621` |
| **9** | **`climatology_baseline`** | Physical Baseline | `0.0714` | `0.36106` | `0.84787` | `0.44159` | `0.81209` | `0.51014` | `0.66319` | `0.41448` | `0.06489` | `0.24659` | `0.04637` |

---

## 15. Feature-Group Ablation Analysis

| Ablation Configuration | Features ($K$) | Val PR-AUC | Val Brier | Holdout Test PR-AUC | Holdout ROC-AUC | Holdout F1 | Holdout Brier | $\Delta$ Val PR-AUC | $\Delta$ Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `all_features` | 28 | `0.67518` | `0.04509` | `0.80202` | `0.98084` | `0.71521` | `0.04155` | `0.00000` | `0.00000` |
| `without_thermo_vpd` | 17 | `0.67733` | `0.04833` | `0.79752` | `0.97956` | `0.72121` | `0.04370` | `+0.00215` | **`-0.00450`** |
| `without_precip_drought` | 22 | `0.67618` | `0.04730` | `0.80496` | `0.98098` | `0.71915` | `0.04271` | `+0.00100` | `+0.00294` |
| `without_radiation` | 27 | `0.67527` | `0.04626` | `0.80311` | `0.98097` | `0.72128` | `0.04172` | `+0.00009` | `+0.00109` |
| `without_antecedent_fire` | 25 | `0.53073` | `0.04796` | `0.70832` | `0.97329` | `0.60777` | `0.03938` | **`-0.14445`** | **`-0.09370`** |
| `without_spatial_seasonal` | 21 | `0.62698` | `0.06512` | `0.75091` | `0.96094` | `0.70391` | `0.07023` | **`-0.04820`** | **`-0.05111`** |

---

## 16. Leave-One-Ecoregion-Out (`LOEO`) Spatial Holdout Generalisation

Each row below holds out the entire target ecoregion from `2001–2010` training and `2011–2015` validation and removes `latitude` and `longitude` so the model must generalize to unseen geography using strictly physical weather and land-cover features:

| Held-Out Unseen Ecoregion | Holdout Test Samples ($N$) | Positive Rate | LOEO Test PR-AUC | LOEO Test ROC-AUC | LOEO Test F1 | LOEO Test Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `central_indian_highlands_forest` | `14,003` | `34.843%` | **`0.78910`** | `0.89695` | **`0.73977`** | `0.14874` |
| `northeast_hills_forest` | `14,915` | `22.689%` | **`0.73087`** | `0.91974` | `0.73891` | `0.09194` |
| `western_ghats_forest` | `8,669` | `11.824%` | **`0.73062`** | `0.95458` | `0.66108` | `0.05804` |
| `eastern_ghats_forest` | `9,026` | `11.677%` | **`0.72968`** | `0.95384` | `0.61798` | `0.06214` |
| `deccan_peninsular_mosaic` | `83,246` | `0.340%` | **`0.71242`** | `0.98179` | `0.70175` | `0.00242` |
| `andaman_nicobar_evergreen_forest` | `5,154` | `0.097%` | `0.55649` | `0.99709` | `0.50000` | `0.00063` |
| `western_himalayas_forest` | `12,267` | `4.247%` | `0.54147` | `0.95298` | `0.52063` | `0.02759` |
| `eastern_himalayas_forest` | `2,720` | `1.875%` | `0.32929` | `0.93254` | `0.31429` | `0.01537` |

---

## 17–21. Error Analysis, Limitations, Conclusions & Verification Suite (`7 / 7 PASSED`)

| Test ID | Verification Test | Status | Verified Output |
| :---: | :--- | :---: | :--- |
| `UT-01` | Tetens VPD & Fosberg FWI physical monotonicity | **PASSED** | `es(0C)=0.6108 kPa, es(30C)=4.2429 kPa` |
| `UT-02` | ERA5 accumulated flux de-accumulation & `m->mm` conversion | **PASSED** | `Hourly increments=[1.0, 2.5, 0.5] mm, daily sum=4.00 mm` |
| `UT-03` | Multi-criteria forest vs. agricultural vs. uncertain attribution | **PASSED** | `['forest_fire', 'agricultural_fire', 'other_or_uncertain']` |
| `UT-04` | Leak-free 72h target alignment (`[D+1..D+3]` vs `<=D` lags) | **PASSED** | `Positive 72h target indices=[3, 4, 5], lag_3d=[0, 0, 0]` |
| `UT-05` | PyTorch `ConvLSTM` / `ConvGRU` / `U-Net` optimizer step & masked focal loss | **PASSED** | `Verified non-zero parameter updates for all 3 architectures` |
| `UT-06` | Spatiotemporal block-bootstrap CI cluster resampling | **PASSED** | `spatiotemporal_block_bootstrap (2deg_lat x 2deg_lon x 7day_window)` |
| `IT-01` | Real ERA5 Zarr vs. manual GRIB message decode verification | **PASSED** | `Month=1947-01, day=1947-01-02, all_passed=True (max_abs_diff=0.0)` |
