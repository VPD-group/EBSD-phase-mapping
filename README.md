# EBSD Phase Mapping Using Deep Learning

This repository contains the code and scripts developed by the VPD Group for **phase mapping in Electron Backscatter Diffraction (EBSD)** data using deep learning approaches, presented at the **Hackathon 2025**.

The project focuses on comparing **standard supervised CNN training (cross-entropy loss)** with **supervised contrastive learning** for EBSD phase classification and mapping of fcc Ni, L1₂ Ni₃Al, and L1₂ Ni₃Fe phases.

---

## Project Overview

Electron Backscatter Diffraction (EBSD) is widely used for crystallographic phase identification, but conventional indexing approaches can struggle with noisy patterns, complex phase mixtures, or limited training data.

In this project, we explore deep learning–based methods to:
- Classify EBSD Kikuchi patterns into crystallographic phases
- Compare cross-entropy loss versus contrastive learning
- Evaluate performance for EBSD phase mapping applications

---

## Methods Implemented

- Convolutional Neural Networks (CNNs)
- EfficientNet architecture
- Supervised learning with:
  - Cross-entropy loss
  - Supervised contrastive loss
- Data augmentation (random horizontal and vertical flips)

---
