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
