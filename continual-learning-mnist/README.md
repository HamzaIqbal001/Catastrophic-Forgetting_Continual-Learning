**_Continual Learning: An Empirical Study of Catastrophic Forgetting_**

**Overview:**
This project investigates catastrophic forgetting in sequential task learning using MNIST digit splits (0–4 → 5–9). The goal is to understand how memory size and model capacity influence knowledge retention under fixed-capacity neural networks.

**The study evaluates:**

- Severity of forgetting under naive sequential training
- Effectiveness of replay-based mitigation
- Relationship between replay buffer size and retention
- Impact of model capacity on forgetting

**Experimental Setup:**

Dataset: MNIST
Task A: Digits 0–4
Task B: Digits 5–9
Training Protocol: Sequential learning
Mitigation Method: Replay buffer

**Metrics Used:**

1. Accuracy after Task A
2. Accuracy after Task B
3. Forgetting
4. Backward Transfer (BWT)

Forgetting is defined as:
Forgetting = Accuracy (after A) - Accuracy (after B)

**Key Findings:**

1. Sequential training without memory causes severe catastrophic forgetting.
2. Replay buffers significantly reduce forgetting.
3. Forgetting decreases nonlinearly as memory size increases.
4. Model capacity interacts with memory size — smaller models exhibit higher forgetting under limited replay.
5. Sufficient replay memory can partially compensate for reduced model capacity.

**Results:**

All experiments are automatically logged to:
results/metrics.csv

Forgetting vs buffer size plot:
results/forgetting_vs_buffer.png

**Project Structure:**

continual-learning-mnist/
│
├── data.py
├── models.py
├── train.py
├── experiment.py
│
├── results/
│ ├── metrics.csv
│ └── forgetting_vs_buffer.png
│
├── requirements.txt
└── README.md

**How to Run:**

pip install -r requirements.txt
python experiment.py

**Future Directions:**

1. Implement regularization-based methods (e.g., EWC)
2. Extend to multi-task continual learning
3. Analyze gradient interference across tasks
4. Study optimization dynamics behind forgetting
5. Explore parameter isolation approaches

**Motivation:**

Understanding the stability–plasticity tradeoff is central to continual learning research. This project serves as a controlled empirical study exploring how memory constraints and model capacity influence catastrophic forgetting.
