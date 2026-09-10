# Understanding Musical Context Using a Hybrid BERT and Graph Neural Network Framework

## Overview

This project explores multimodal music understanding by combining:

- **BERT** for semantic understanding from music descriptions
- **Graph Neural Networks (GNNs)** for structural music representation learning
- **Multimodal fusion methods** for combining textual and graph-based information

The objective is to investigate whether combining semantic information from text with structural information from music representations can improve music context understanding.

---

## Project Tasks

### Task 1: BERT-based MusicCaps Multi-label Classification

A BERT-based classifier is trained on the MusicCaps dataset to understand musical descriptions and predict multiple musical attributes.

### Task 2: GNN-based Music Structural Learning

Audio representations are converted into graph structures where nodes represent musical components and edges represent relationships between them.

### Task 3: BERT-GNN Multimodal Fusion

Different fusion strategies are evaluated:

- Feature concatenation
- Gated fusion
- Cross-attention fusion

---

# Results

## Task 1: BERT MusicCaps Classification

| Metric | Score |
|---|---|
| Micro-F1 | 0.7001 |
| Macro-F1 | 0.5701 |
| Macro AUC-PR | 0.7075 |

The results demonstrate that BERT can effectively capture semantic information from human-written music descriptions.

---

## Task 2 & Task 3: FMA Classification

| Model | Accuracy | Macro-F1 | AUC-PR |
|---|---|---|---|
| BERT Only | 37.26% | 32.99% | 32.84% |
| GNN Only | 55.03% | 44.80% | 47.63% |
| Cross Attention Fusion | **55.25%** | **45.31%** | **47.72%** |

The Cross Attention Fusion model achieved the best overall performance by learning interactions between semantic and structural representations.

---

# Project Structure
