# 🧬 Leukemia Patient Classifier

This project implements a supervised classification pipeline using Support Vector Machines (SVMs) to distinguish between two subtypes of leukemia patients based on gene expression profiles. The dataset includes expression levels of 2,000 genes across 79 patients, labeled as either cytogenetically normal (-1) or with a chromosomal translocation (+1).

## 📂 Project Structure

- `Homework 03.pdf/`: The homework given by the professors of the course
- `homework_3.ipynb/`: Jupyter notebook containing all experiments and results
- `images/`: Plots and figures used in the report (ROC, confusion matrices, etc.)
- `gene_expr.tsv/`: dataset
- `Leukemia.md`: Final report (generated from the notebook, to visualize it download also the images)

## 🔍 Objectives

- Train and evaluate SVM classifiers with different kernels:
  - Linear
  - Polynomial
  - RBF (Gaussian)
- Optimize hyperparameters via cross-validation
- Evaluate models with accuracy, ROC AUC, precision, recall, and F1-score
- Repeat the analysis using a filtered dataset containing only the top 5% most variable genes

## 📊 Dataset

The dataset contains:
- 2,000 gene expression features
- 79 patients
- Binary outcome (`y`):  
  - `+1`: chromosomal translocation  
  - `-1`: cytogenetically normal

## 🧪 Technologies

- Python 3.11
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
- Jupyter Notebook / Quarto (for report rendering)
- Markdown

## 📄 Report

The full report is available as `Leauekimia.md`, and includes:
- Model descriptions and hyperparameter tuning
- ROC curves and confusion matrices
- Comparative performance analysis
- Discussion and conclusions
