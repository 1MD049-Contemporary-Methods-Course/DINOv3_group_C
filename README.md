# Task 3: Depth Estimation with DINOv3

**Authors:** Junming Ma, Petter Möllerström, Xingping Lyu

## Overview
This repository implements Monocular Depth Estimation using the **DINOv3** Vision Transformer as a backbone. It provides a complete pipeline for environment setup, weight acquisition, and inference on the NYU Depth V2 dataset.

## File Structure
* **`task3.ipynb` (Main Demo):** The primary entry point. This notebook:
    * Installs dependencies and configures the environment.
    * Auto-patches necessary files.
    * Downloads the fine-tuned model (`hybrid_best.pth`) and dataset.
    * Performs inference to visualize RGB inputs and predicted depth maps.
* **`depth.ipynb` (Training & Dev):** The development notebook. It contains:
    * The training loop.
    * Model architecture definitions (utilizing LoRA and quantization).
    * Validation metrics.

## Architecture
* **Backbone:** DINOv3 (ViT-L/16)
* **Decoder:** DPT (Dense Prediction Transformer) Head
* **Optimization:** Trained using SiLogLoss with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## Getting Started
To run the demo:
1.  Open **`task3.ipynb`**.
2.  Run all cells sequentially. The notebook handles all downloads and path configurations automatically.

## Requirements
* Python 3.10+
* PyTorch (CUDA recommended)
* Transformers
* Hugging Face Hub
* Torchmetrics
