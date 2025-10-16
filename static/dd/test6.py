import matplotlib.pyplot as plt

# Sample performance metrics (replace with actual values)
metrics = {
    'Accuracy': 0.89,
    'Precision': 0.85,
    'Recall': 0.87,
    'F1-Score': 0.86,
    'AUC-ROC': 0.90
}

# Unpack metric names and values
labels = list(metrics.keys())
values = list(metrics.values())

# Create bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color='mediumseagreen')

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

# Graph details
plt.ylim(0, 1.1)
plt.title('Performance Metrics of Adversary Detection Module')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('static/graph5.png')
plt.show()
