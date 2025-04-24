# 🌀 Hybrid Diffusion Ultrasound

A research-driven project leveraging **diffusion models** and **low-rank adaptation techniques (LoRA)** to generate synthetic ultrasound images, particularly targeting **class-imbalanced medical datasets** (e.g., BUSI).

> This project explores the impact of generative augmentation methods such as **LoRA**, **Textual Inversion**, **ControlNet**, and **Img2Img** for enhancing minority class representation in ultrasound datasets — with applications in training more balanced classifiers.

---

## 🧠 Core Features

- ✅ Fine-tuning Stable Diffusion with **LoRA**
- ✅ Hybrid augmentation using **LoRA + Textual Inversion**
- ✅ Controlled generation via **ControlNet** and refine using **Image-to-Image**
- ✅ Balanced dataset generation for classification
- ✅ Evaluation using **FID**
- ✅ Training baseline classifier (e.g., **ResNet18**) on real + synthetic data

---

## 📁 Directory Structure

```bash
.
├── data/                          # Ignored: contains real/synthetic images
├── diffusers/                    # Custom/forked Diffusion model (LoRA-compatible)
├── results/
├── generate_batch_image_*.ipynb  # Generation pipelines using LoRA, TI, ControlNet
├── evaluate_gens.ipynb          # Evaluates generated images (FID, SSIM, etc.)
├── train_resnet18.ipynb         # Classifier training notebook
├──  create_dataset_balanced.ipynb# Merges real + synthetic data
├── scripts/
├── train_text_to_image_lora.sh
├── train_textual_inversion.sh
├── .gitignore
├── requirements.txt
└── README.md
```
---

## 🚀 Setup
### 1. Clone the repository

```bash
git clone https://github.com/farhanfuadabir/hybrid_diffusion_ultrasound.git
cd hybrid_diffusion_ultrasound
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## 📦 Usage

### 🖼️ Generate Synthetic Images

Run any of the generation notebooks:

- `generate_batch_image_LoRA.ipynb`
- `generate_batch_image_LoRA_TI_controlNet.ipynb`

Each notebook targets a different combination of augmentation methods.

### 🧪 Evaluate Generated Data

Use:

- `evaluate_gens.ipynb` for FID or visual comparisons.

### 🧠 Train Classifier

Dataset:

- Download the BUSI dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
- Extract the dataset under `./data/BUSI` directory.
- Run `process_data.ipynb` to generate the captions.
- Run `process_edges.ipynb` to generate the edges from the edges.

Run:

- `train_text_to_image_lora.sh` to fine-tune the text-to-image stable diffusion v1.5 with LoRA.
- `train_textual_inversion.sh` for token embedding.
- `train_resnet18.ipynb` to evaluate model performance on real vs hybrid datasets.
---

