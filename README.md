# Visualise-GPR: Gaussian Process Surrogate Analysis for CAFM

An interactive analysis and visualization tool suite for sweeping partial dependencies and evaluating Gaussian Process Regression (GPR) models on Conductive Atomic Force Microscopy (CAFM) pedestrian simulation datasets.

## Architecture Overview

\\\
User Interface / Analyst
    │
    ▼
┌──────────────────────────────────────────────┐
│  Configuration (Dataset, Target, Swipes)     │  ← Defines frozen parameters & target metric
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Data Loader & Preprocessor                  │  ← Loads .npz, splits data, standardizes
└──────────┬───────────────────────────────────┘
           │ (Standardized Tensors)
           ▼
┌──────────────────────────────────────────────┐
│  Gaussian Process Regressor (scikit-learn)   │  ← Fits Kernel (Matern, RBF, RationalQuadratic)
│  - Optimizes Hyperparameters (L-BFGS-B)      │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Partial Dependence Sweep Generator          │ ← Freezes 5/6 dimensions, sweeps the target
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│  Interactive Visualizer (HTML/JS + Python)   │ ← Plots Uncertainty (±2 std dev), Mean, Parity
└──────────────────────────────────────────────┘
\\\

## System Output

The system outputs statistical fits and graphical plots mapping 6D input spaces onto physical metrics:

| Metric Evaluated | GP Prediction Mean | Confidence Interval (±2σ) | Kernel Extracted |
|---|---|---|---|
| Mean Crossing Time | 4.25 s | ±0.15 s | Matern(nu=2.5) |
| Flow Rate | 2.10 ped/s | ±0.30 ped/s | RBF + WhiteKernel |

It also generates an interactive Web UI (\index.html\) to scrub through these multi-dimensional sweeps smoothly in the browser.

## Directory Structure

\\\
├── gp_cafm_analysis.py         # Core Python ML script for data loading, GP fitting, and sweeps
├── gp_cafm_visualizer.html     # Interactive Web UI for plotting Gaussian Process results
├── index.html                  # Alias/Symlink to the visualizer for direct web hosting
└── README.md                   # Project documentation
\\\

## How to Run

### 1. Prerequisites
- Python 3.9+
- Modern Web Browser
- Libraries: \
umpy\, \scikit-learn\, \matplotlib\, \scipy\

### 2. Install Dependencies

\\\ash
pip install numpy scikit-learn matplotlib scipy
\\\

### 3. Run the Python Analysis

\\\ash
# Edit the CONFIG block in gp_cafm_analysis.py to point to your .npz dataset
python gp_cafm_analysis.py
\\\

### 4. Run the Visualizer
Simply open \index.html\ or \gp_cafm_visualizer.html\ in any web browser to view the interactive JS-driven plots of the Gaussian Process. No web server is strictly required for local viewing.

## Key Design Decisions

1. **Frozen-Dimension Sweeps**: To visualize a 6D parameter space (v0_mean, tau, A, R_safety, density, flow_ratio), the system mathematically freezes 5 dimensions (either at their statistical mean or randomly bounded) and sweeps the 6th. This creates digestible 2D slices of the surrogate model's behavior.
2. **Multiple Kernel Benchmarking**: The code natively supports importing and testing various kernels (Matern, RBF, RQ, ExpSineSquared) since pedestrian fluid dynamics exhibit both smooth and periodic chaotic properties.
3. **Decoupled Frontend**: The heavy GPR mathematical lifting is done in Python, while the visualization is exported/served in a standalone HTML file. This prevents analysts from needing Jupyter running just to share a plot with stakeholders.
