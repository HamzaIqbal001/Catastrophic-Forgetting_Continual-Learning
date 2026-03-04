# Continual Learning: Empirical Study of Catastrophic Forgetting

## Overview

This repository presents a controlled empirical study of catastrophic forgetting in sequential task learning. Using MNIST digit splits (0–4 → 5–9), the project investigates how replay memory size and model capacity influence knowledge retention in fixed-capacity neural networks.

The study analyzes the stability–plasticity tradeoff through systematic experimentation and formal continual learning metrics.

---

## Research Questions

1. How severe is catastrophic forgetting under naive sequential training?
2. How effectively does replay mitigate forgetting?
3. How does replay buffer size influence retention?
4. How does model capacity interact with memory size?

---

## Experimental Setup

Dataset: MNIST  
Task A: Digits 0–4  
Task B: Digits 5–9  
Training Protocol: Sequential learning  
Mitigation Strategy: Fixed-size replay buffer

### Models Evaluated

- SimpleCNN (Higher Capacity)
- SmallCNN (Reduced Capacity)

---

## Evaluation Metrics

### Forgetting

Forgetting = Accuracy(A after A) − Accuracy(A after B)

Measures how much Task A performance drops after learning Task B.

### Backward Transfer (BWT)

BWT = Accuracy(A after B) − Accuracy(A after A)

Interpretation:
- Negative → Forgetting
- 0 → Perfect retention
- Positive → Positive transfer

Both metrics are recorded for all experiments.

---

## Key Findings

- Sequential training without memory leads to severe catastrophic forgetting.
- Replay significantly improves knowledge retention.
- Forgetting decreases as replay memory increases.
- Smaller models tend to forget more under limited memory.
- Increasing replay memory can partially compensate for smaller model capacity.

---

## Results

All experiments are logged in:

results/metrics.csv

Generated visualizations:

results/forgetting_vs_buffer.png  
results/bwt_vs_buffer.png

---

## Project Structure

continual-learning-mnist/

    data.py
    models.py
    train.py
    experiment.py

    results/
        metrics.csv
        forgetting_vs_buffer.png
        bwt_vs_buffer.png

    requirements.txt
    README.md

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Running the Experiment

python experiment.py

This will:

- Train models sequentially
- Sweep replay buffer sizes
- Log experiment metrics
- Generate evaluation plots automatically

---

## Future Work

- Implement regularization-based continual learning methods (EWC, SI)
- Extend experiments to multiple sequential tasks
- Analyze gradient interference during training
- Investigate optimization dynamics behind forgetting
- Explore parameter isolation methods

---

## Motivation

Continual learning systems must balance stability (retaining old knowledge) with plasticity (learning new tasks). This project provides an empirical analysis of how memory constraints and model capacity influence catastrophic forgetting, serving as a foundation for further research in continual learning systems.
