# slc-localization
The codebase created to support the application of artificial intelligence for predicting the subcellular localization of solute carrier transporter (SLC) proteins. We developed an iterative method that harmonizes human annotations with AI-based model outputs.
A robust, modular pipeline for end-to-end SLC (solute carrier) image analysis, including data download, embedding generation, model training, and compartment-specific reporting.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Data Download](#data-download)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs & Results](#outputs--results)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Contact](#contact)

---

## Overview
This project provides a complete workflow to:
- Download and validate large-scale imaging data
- Generate image embeddings using a pre-trained model
- Train and evaluate models for SLC compartment classification
- Produce detailed reports and summary statistics

The pipeline is modular, robust to interruptions, and easy to resume.

## Project Structure
```
├── data/                  # Raw and processed data, including images and results
│   ├── file_download.sh   # Robust shell script for downloading images
│   └── ...
├── src/                   # Source code
│   ├── data/              # Data processing and embedding generation
│   ├── models/            # Model definitions
│   └── training/          # Training and evaluation scripts
├── main.py                # Main entry point for the pipeline
├── pyproject.toml         # Python dependencies
├── README.md              # This file
```

## Prerequisites & Environment Setup
- Python 3.10+
- Recommended: Linux/macOS with bash/zsh shell
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

**Install dependencies using uv:**
```bash
# Install uv if not already installed (standalone installer; see
# https://docs.astral.sh/uv/getting-started/installation/ for alternatives)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install all dependencies from pyproject.toml
uv pip install .
```

## Data Download
1. **Prepare the file list:** Place your TSV file (e.g., `filelist_sample_HATag.tsv`) in the `data/` directory.
2. **Run the download script:**
   ```bash
   cd data
   bash file_download.sh 0 1000  # Download first 1000 files (adjust as needed)
   ```
   - The script is robust: it skips existing files, retries failed downloads, and validates images.
   - To resume or download a different range, adjust the start/end row arguments.
   - For large downloads, use `screen` or `tmux` to avoid interruption.
3. **Download annotated data:**
   - Downloaded the data directly from the Resolute website: https://dataresolute.blob.core.windows.net/public/annotation/SLC_localization.xlsx

## Running the Pipeline
1. **Activate your environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Run the main analysis:**
   ```bash
   python main.py
   ```
   - This will:
     - Generate image embeddings
     - Save `embeddings.csv` and `file_list.csv`
     - Run compartment analysis and save results in `data/compartment_results/`

3. **Customizing analysis (CLI flags):**

   The pipeline is configured via command-line flags (run `python main.py --help`
   for the full list):

   | Flag | Default | Description |
   | --- | --- | --- |
   | `--compartments` | `"Plasma membrane"` | One or more compartments to analyze. Use `--list_compartments` to see all options. |
   | `--seeds` | `10 42 123` | Random seeds for multi-seed evaluation; metrics are reported as mean ± std across seeds. |
   | `--batch-size` | `32` | Batch size for embedding extraction (DenseNet forward pass). |
   | `--num-workers` | `4` | DataLoader worker processes for embedding extraction. **Use `0` as a safe fallback on macOS/Windows.** |
   | `--data_dir` | `./data` | Base data directory. |
   | `--output_dir` | `<data_dir>/compartment_results` | Where to write per-compartment results. |

   Example:
   ```bash
   python main.py --compartments "Plasma membrane" --seeds 10 42 123 \
       --batch-size 32 --num-workers 4
   ```

   List all available compartments and exit:
   ```bash
   python main.py --list_compartments
   ```

## Outputs & Results
- `embeddings.csv`: Image embeddings for all processed images
- `file_list.csv`: List of image file paths
- `data/compartment_results/`: Contains per-compartment reports, classification metrics, and summary tables
  - `compartment_summary.csv`: Per-compartment metrics aggregated across seeds as mean ± std (`roc_auc_mean`/`roc_auc_std`, plus PR-AUC, F1, precision, and recall).
  - `<compartment>_results.csv`: Long-format per-gene predictions with one row per `(gene, seed)`.

## Troubleshooting & Tips
- **Resuming downloads:** The shell script skips files that already exist and only counts valid images.
- **Session persistence:** For long downloads, use `screen` or `tmux` to avoid losing progress if your terminal disconnects.
- **Missing dependencies:** Reinstall the project into your environment with `uv pip install .` (all dependencies are declared in `pyproject.toml`).
- **DataLoader worker crashes:** On macOS/Windows, if embedding extraction hangs or errors with a multiprocessing/worker message, rerun with `--num-workers 0`.
- **Custom data:** Use `--data_dir` / `--output_dir` (and the other flags above) to point the pipeline at your data locations.

## Contact
For questions or support, please open an issue or contact the project maintainer.

---

<small>Project organized according to the <a target="_blank" href="https://github.com/LeanderK/cookiecutter-ml">cookiecutter machine learning template</a>.</small>