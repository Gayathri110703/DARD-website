import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Simulated thresholds
thresholds = np.linspace(0.1, 0.9, 9)

# Simulated performance metrics (you can replace these with real values)
precision = [0.60, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89]
recall =    [0.92, 0.90, 0.88, 0.84, 0.80, 0.76, 0.72, 0.70, 0.68]
f1_score =  [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
accuracy =  [0.75, 0.78, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89]

# Set up plot style
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision, label='Precision', marker='o')
plt.plot(thresholds, recall, label='Recall', marker='o')
plt.plot(thresholds, f1_score, label='F1-Score', marker='o')
plt.plot(thresholds, accuracy, label='Accuracy', marker='o')

# Labels and Title
plt.title('Performance Metrics of Adversary Detection Module')
plt.xlabel('Anomaly Detection Threshold')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(thresholds)
plt.legend()
plt.tight_layout()
plt.show()
