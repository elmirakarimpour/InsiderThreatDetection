# Unsupervised Insider Threat Detection Using BiLSTM-Based Deep Clustering

**Master's Thesis Research Project | Shiraz University**

**Status:** Work in Progress

## Overview

Insider threat detection is a challenging problem in cybersecurity because malicious activities are rare, may closely resemble legitimate user behavior, and are often represented by a limited number of reliable examples.

This research investigates an unsupervised deep learning framework for identifying suspicious behavioral patterns from sequential user activity data. The central objective is to learn informative behavioral representations that capture temporal dependencies and user-specific characteristics, enabling the identification of deviations from normal behavior without relying on predefined attack categories.

## Research Objectives

The main objectives of this research are to:

- Learn meaningful representations from sequential user activity data.
- Capture temporal dependencies and user-specific behavioral characteristics.
- Investigate the role of contrastive learning in improving representation quality.
- Explore deep clustering for identifying deviations in the learned representation space.
- Evaluate how model architecture, learning objectives, and data imbalance influence detection performance and representation stability.

## Methodological Approach

The proposed framework combines the following components:

**1. Sequential Behavioral Modeling**

A Bidirectional Long Short-Term Memory (BiLSTM) Autoencoder is used to model user activity sequences and learn behavioral representations.

**2. User-Specific Representations**

Learnable user embeddings are incorporated to capture individual behavioral characteristics and contextual differences among users.

**3. Contrastive Learning**

A contrastive learning objective is integrated to encourage the learning of informative and discriminative behavioral representations.

**4. Deep Clustering**

Deep clustering is employed to organize the learned representation space and support the identification of deviations from normal behavioral patterns.

## Evaluation Strategy

The framework is evaluated through a structured experimental analysis, including:

- Comparisons with conventional machine-learning and deep-learning baselines.
- Ablation studies to investigate the contribution of individual components.
- Repeated experiments with different random seeds to examine result stability.
- Analysis of the effects of data imbalance and modeling choices on detection behavior.

## Implementation

The research framework is being developed primarily using **PyTorch**, with **scikit-learn** used for data analysis, evaluation, and comparisons with conventional machine-learning methods.

The implementation is organized around the main stages of sequential data modeling, representation learning, clustering, and experimental evaluation.

## Repository Status

This repository contains the ongoing implementation of my Master's thesis research.

The framework, experimental pipeline, and documentation are currently under development. Individual components and experimental configurations may be updated as the research progresses. Complete implementation details and validated experimental results will be documented as the work advances.

**Please note:** This repository represents an ongoing research project and should not yet be considered a finalized or fully reproducible software release.

## Research Output

A manuscript based on this research is currently in preparation:

*Unsupervised Insider Threat Detection Using BiLSTM-Based Deep Clustering*

## Technologies

- Python
- PyTorch
- scikit-learn
- NumPy
- Pandas

