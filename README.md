# Spatial ECM Feature Extraction

Computational pipeline for quantifying extracellular matrix (ECM) organization from multiplex tissue imaging to identify structural biomarkers associated with the tumor microenvironment.

---

## Overview

The extracellular matrix plays an important role in regulating immune infiltration, tumor progression, and therapeutic response. This project develops a computational workflow for extracting quantitative collagen features from multiplex tissue imaging and identifying biologically meaningful ECM signatures.

Using whole-core tissue images, this pipeline performs automated feature extraction, feature reduction, and statistical analysis to identify ECM characteristics that may contribute to tumor behavior and patient outcomes.

---

## Biological Question

Can quantitative collagen architecture reveal spatial biomarkers associated with the tumor microenvironment and future clinical outcomes?

---

## Computational Pipeline

Raw Histocat Images

↓

Collagen Feature Extraction

↓

Feature Engineering

↓

Correlation Analysis

↓

Feature Selection

↓

Candidate ECM Biomarkers

---

## My Contributions

- Developed an automated Python pipeline for quantitative collagen feature extraction from multiplex tissue imaging
- Engineered morphological, intensity, texture, and structural ECM features across tissue cores
- Built computational workflows for feature reduction using correlation analysis and redundancy filtering
- Generated concise feature sets for downstream statistical modeling and biological interpretation
- Produced publication-ready visualizations including correlation matrices and feature selection analyses
- Organized reproducible analysis notebooks and modular scripts for future expansion to tumor cohorts

---

## Methods

Feature extraction includes:

- Collagen intensity statistics
- Texture analysis
- Morphological measurements
- Fiber organization metrics
- Correlation-based feature reduction
- Statistical visualization
- Candidate biomarker selection

---

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- scikit-image
- OpenCV
- Matplotlib
- Jupyter Notebook

---

## Current Status

Current analyses focus on normal tissue cores for computational feature development.

Future work will apply the optimized feature set to tumor specimens to investigate relationships between ECM organization, immune infiltration, and clinical outcomes.

---

## Poster

Preliminary work presented at:

**University of Maryland Summer Undergraduate Research Conference (SURC), 2026**

Poster available in the `results/` directory.

---

## Author

**Arushi Verma**

Bioengineering • Computational Biology • Spatial Omics • Translational Research
