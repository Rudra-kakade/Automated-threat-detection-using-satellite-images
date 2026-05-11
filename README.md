# Automated Threat Detection Using Satellite Images

This project implements a satellite-based threat detection system using YOLOv8-OBB (Oriented Bounding Boxes). It includes modules for data preparation (tiling and augmentation), model training, evaluation, and inference with SAHI (Slicing Aided Hyper Inference).

## Features

- **Data Tiling**: Efficiently process large satellite images into manageable tiles.
- **YOLOv8-OBB Training**: Specialized training for oriented objects in satellite imagery.
- **Advanced Evaluation**: Comprehensive metrics including confusion matrices and PR curves.
- **SAHI Inference**: Optimized detection for small objects in large-scale images.

## Project Structure

- `src/data`: Data preparation and augmentation scripts.
- `src/train`: Model training logic.
- `src/eval`: Evaluation and visualization tools.
- `src/inference`: Inference and coordinate verification.
- `tests`: Unit tests for core components.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Training**: `python src/train/train.py --data src/data/dataset.yaml`
- **Evaluation**: `python src/eval/evaluate.py --model path/to/best.pt`
- **Inference**: `python src/inference/infer_sahi.py --source image.png`
