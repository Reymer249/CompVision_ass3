# Gesture Classification using ResNeXt-101 Features

This repository contains the implementation of a gesture classification pipeline based on **segment-based frame sampling** and **pretrained ResNeXt-101 feature extraction**, followed by training lightweight MLP classifiers.  
The work was done as part of a Computer Vision assignment at Leiden University.

For a detailed explanation of the methodology, experiments, and results, please see the [report.pdf](./report.pdf).

---

## 📖 Project Overview

- **Task**: Hand gesture classification using the [Jester dataset](https://www.qualcomm.com/developer/software/jester-dataset).  
- **Approach**:
  - Videos are divided into temporal segments.
  - A single frame is sampled from each segment.
  - Features are extracted using a **ResNeXt-101** CNN pretrained on ImageNet.
  - Features are fed into a **Multi-Layer Perceptron (MLP)** for classification.
- **Key Findings**:
  - Smaller MLPs perform better than larger ones.
  - Equidistant (first frame) sampling outperforms random uniform sampling.
  - Reducing segments from 8 → 4 halves training time with minimal accuracy loss.

---

## 🧪 Models & Experiments

Six models were trained, varying in MLP architecture, frame sampling, and number of segments.

| Model | Params (MLP) | Frame Selection | Segments | Test Accuracy | Test Loss |
|-------|--------------|-----------------|----------|---------------|-----------|
| 1     | 75.5M        | Random (uniform) | 8        | 0.499         | 1.708     |
| 2     | 257.3M       | Random (uniform) | 8        | 0.456         | 1.783     |
| 3     | 8.4M         | Random (uniform) | 8        | 0.531         | 1.742     |
| 4     | 8.4M         | First frame      | 8        | 0.546         | 1.796     |
| 5     | 4.2M         | First frame      | 4        | 0.537         | 1.714     |
| 6     | 2.1M         | First frame      | 2        | 0.433         | 1.995     |

Plots of training time, accuracy, and loss can be found in the [`plots`](./plots) directory.


---

## 📂 Repository Structure

- `helpers_scripts/`
  - `split_train_val.py` – Script to split dataset into training and validation sets
  - `test.ipynb` – Notebook for evaluating trained models
- `logs/`
  - `train_model1.log` – Training log for Model 1
  - `train_model2.log` – Training log for Model 2
  - `train_model3.log` – Training log for Model 3
  - `train_model4.log` – Training log for Model 4
  - `train_model5.log` – Training log for Model 5
  - `train_model6.log` – Training log for Model 6
- `metrics/`
  - `model_1/`
    - `accuracies.pkl` – Accuracy values across training
    - `losses.pkl` – Loss values across training
  - `model_2/` … `model_6/` – Same structure as `model_1/`
  - `test_acc.pkl` – Final test accuracies
  - `test_loss.pkl` – Final test losses
  - `time_taken.pkl` – Training times (approx., extracted from logs)
- `models_training_code/`
  - `model1.py` – Training script for Model 1
  - `model2.py` ... `model6.py` – Training scripts for Models 2-6
  - `model_1.ipynb` – Notebook version of Model 1 training
  - `model_2.ipynb` … `model_6.ipynb` – Notebooks for Models 2–6
- `plots/`
  - `Plots.ipynb` – Notebook to generate result plots
  - `accuracies_plot.png` – Accuracy curves
  - `losses_plot.png` – Loss curves
  - `time_bar.png` – Training time comparison
- `splits/`
  - `jester-v1-labels.csv` – Gesture class labels
  - `jester-v1-train.csv` – Original training split (v1)
  - `jester-v1-validation.csv` – Original validation split (v1)
  - `train.csv` – Training set
  - `val.csv` – Validation set
  - `test.csv` – Test set
- `report.pdf` – Full project report
- `README.md` – Project documentation (this file)

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas, numpy, matplotlib
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
