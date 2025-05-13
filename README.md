# 🚦 LUR Space State Fusion Fusion Pipeline

---

## 📘 About

This project applies **computational statistics** and **probabilistic programming** to infer and forecast urban air pollution using existing open data. We combine satellite observations (TROPOMI), traffic volumes (SCATS), and land-use features (OSM) into a modular pipeline powered by **Bayesian state space models** and **Land Use Regression (LUR)**.

Our focus is on scalable, uncertainty-aware modeling using **PyMC**, enabling high-resolution (100m × 100m) pollution estimates without dense sensor networks.

---

This repository provides a modular pipeline to estimate and forecast urban air pollution (e.g., NO₂) using data fusion techniques across multiple sources:

* **TROPOMI satellite scenes** (via Copernicus/Sentinel-5P)
* **SCATS traffic volume data**
* **OpenStreetMap-derived land-use features**
* **Land Use Regression + State Space models**

The goal is to produce a high-resolution, 100m × 100m gridded dataset of inferred and forecasted pollution values for urban planning, environmental research, and real-time alerting systems.

---

## 🧭 Key Features

* ✅ **Check-before-download logic** for TROPOMI scenes to avoid redundant fetches.
* 📉 **Downsampling TROPOMI scenes** using bilinear interpolation to 100m × 100m.
* 📍 **NO₂ extraction at grid centroids** for direct use in LUR models.
* 🛣️ **Fusion-ready input matrix** combining satellite, traffic, and spatial features.
* 🔮 Supports **Space State Models with Uncertainty Quantification**.
* 🧪 **Unit-tested modules** for reliable integration in larger ML pipelines.

---

## 🗂️ Project Structure

```bash
tropomi-pollution-fusion/
├── data_acquisition/
│   └── tropomi_handler.py       # Check/download/downsample/extract NO₂
├── feature_engineering/
│   ├── grid_generator.py        # Generates 100m × 100m grid from AOI
│   ├── osm_extractor.py         # Extracts OSM features per grid
│   └── traffic_mapper.py        # Maps SCATS counts to grid cells
├── modeling/
│   ├── lur_model.py             # Land Use Regression with covariates
│   ├── state_space_model.py     # Space-Time Forecasting model
│   └── uncertainty.py           # EWFM & UQ integration
├── tests/
│   └── test_*.py                # Unit tests for each module
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   └── run_pipeline.py          # End-to-end data processing CLI
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/your-username/tropomi-pollution-fusion.git
cd tropomi-pollution-fusion
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py --aoi "path/to/aoi.geojson" --start-date 2023-01-01 --end-date 2023-01-31
```

---

## 🧪 Running Unit Tests

```bash
pytest tests/
```

---

## 📄 Methodology

This project supports a methodological framework that includes:

* **Downscaling satellite imagery** using Ensemble Weighted Fusion Models (EWFM)
* **Mapping SCATS traffic volumes** as a proxy for emission intensity
* **Spatiotemporal modeling** using Gaussian Process State Space Models (GPSSM)
* **Uncertainty Quantification** of model predictions and downscaling steps

For details, please refer to the [`docs/methodology.md`](docs/methodology.md) file.

---

## 🧠 Acknowledgements

* Sentinel-5P data courtesy of the Copernicus Program
* SCATS traffic data provided by \Dublin City Council
* OSM data via the Overpass API and `osmnx`

---

## 📜 License

MIT License — see the [LICENSE](LICENSE) file for details.

# lur_spacestatemodel_integration
