import matplotlib.pyplot as plt
import seaborn as sns

# Techniques and their simulated impact on adversary clustering accuracy
techniques = ['Original', 'Basic Shuffle', 'Shuffle Increment', 'Shuffle Reduction', 'Change Topic']
clustering_accuracy = [0.91, 0.65, 0.52, 0.48, 0.35]  # Simulated drop in clustering accuracy

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
bars = plt.bar(techniques, clustering_accuracy, color=['#4caf50', '#2196f3', '#ff9800', '#f44336', '#9c27b0'])

# Annotate bars with accuracy values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

plt.ylim(0, 1.1)
plt.title('Impact of DARD Manipulation Techniques on Clustering Accuracy')
plt.ylabel('Clustering Accuracy (simulated)')
plt.xlabel('Manipulation Technique')
plt.tight_layout()
plt.show()
