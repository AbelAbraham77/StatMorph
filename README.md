# StatMorph

StatMorph is a Streamlit dashboard for morphing 2D datasets into target shapes while preserving key statistical properties, then evaluating the impact on downstream machine learning performance. It wraps the `data-morph-ai` library with a visual, interactive workflow and adds statistical, ML, and clustering diagnostics plus exportable reports.

This project is especially useful for teaching or exploring how different geometric representations can keep means, variances, and correlations intact while dramatically changing visual structure.

**Status**: The app is fully runnable from `app.py`. The repository does not currently include a license file.

## Key Features
- Morph 2D point clouds into many predefined shapes while preserving statistics.
- Built-in starter datasets from `data-morph-ai` (e.g., `dino`, `cat`, `dog`).
- Real-world datasets via public APIs (stock market and global weather).
- Custom shape creator using mathematical or parametric equations.
- Statistical integrity report with KS tests and correlation preservation.
- ML performance comparison across multiple classifiers.
- Clustering diagnostics (K-means inertia and silhouette).
- Export: animated GIF, CSV summaries, and PDF reports.

## Demo Workflow (What You Can Do)
1. Load a dataset (upload, starter set, real-world, or custom equation).
2. Choose a target shape and morphing precision/iterations.
3. Visualize original vs morphed datasets side-by-side.
4. Review statistical integrity and ML preservation metrics.
5. Export reports and data for downstream use.

## Installation
### Prerequisites
- Python 3.9+ recommended
- Internet access if you want real-world datasets

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Run The App
```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.

## Data Sources
StatMorph provides multiple dataset sources:

### 1) Upload Your Own CSV
Your CSV must contain numeric `x` and `y` columns with at least 10 points. Example:
```csv
x,y
1.0,2.0
2.0,3.5
3.5,1.2
```

### 2) Starter Datasets (from `data-morph-ai`)
Available shapes include:
`dino`, `cat`, `dog`, `bunny`, `panda`, `sheep`, `gorilla`, `music`, `pi`, `python`, `soccer_ball`, `superdatascience`.

### 3) Real-World Datasets
The app includes two real-world datasets:
- Stock Market (AAPL): closing price vs trading volume (via Yahoo Finance / `yfinance`)
- World Weather Data: temperature vs humidity (via `wttr.in`)

If those APIs fail or are unavailable, the weather loader falls back to realistic synthetic data.

### 4) Custom Shape Creator
You can generate datasets from:
- Mathematical functions: `y = f(x)` (e.g., `sin(x/10)`, `x**2`)
- Parametric equations: `x = f(t), y = g(t)` (e.g., `x=cos(t)`, `y=sin(t)`)

## Morphing Controls (UI)
The sidebar exposes key controls:
- Target Shape: choose from built-in shape list
- Decimal Precision: lower values allow more visual change
- Iterations: higher values improve shape fidelity (slower)
- Animation: optional morphing GIF with easing and freeze frames

## Statistical & ML Evaluation
StatMorph compares original vs morphed datasets using:
- Means, standard deviations, and correlation
- Kolmogorov-Smirnov tests for distribution similarity
- Basic ML evaluation (Logistic Regression)
- Comprehensive ML evaluation (RF, SVM, KNN, Naive Bayes, Decision Tree, MLP)
- Clustering comparison (KMeans inertia and silhouette)
- Overall preservation score combining stats + ML metrics

## Exports
From the UI you can generate:
- Animated GIF of the morphing process
- CSV summary of stats and ML metrics
- PDF report of the analysis
- Morphed dataset CSV

## Project Structure
```
.
├─ app.py                  # Streamlit UI + orchestration
├─ morph_engine.py         # Wrapper around data-morph-ai morphing
├─ evaluator.py            # Statistical + ML evaluation
├─ real_world_datasets.py  # Public API dataset loaders
├─ custom_shapes.py        # Equation-based custom shape utilities
├─ requirements.txt        # Python dependencies
├─ sample_data.csv         # Example dataset
└─ sample_shape.png        # Example shape image (not currently used in UI)
```

## Notes and Caveats
- Some advanced ML evaluation uses `GridSearchCV` for MLP; this can be slow on large datasets.
- Real-world dataset fetching requires internet access; `yfinance` also requires its API to be reachable.
- No license file is included in this repository.

## Troubleshooting
- If `yfinance` is missing, install it with `pip install yfinance`.
- If Streamlit fails to start, ensure you are in the project directory and using the right Python environment.
- If morphing fails, confirm your dataset has numeric `x`/`y` columns and at least 10 rows.

## Acknowledgements
StatMorph builds on the `data-morph-ai` library for the morphing algorithm and uses standard data science tooling for evaluation.
