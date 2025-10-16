import matplotlib.pyplot as plt
import seaborn as sns

# Techniques
techniques = ['Original', 'Basic Shuffle', 'Shuffle Increment', 'Shuffle Reduction', 'Change Topic']

# Simulated accuracy data for different models under each DARD technique
kmeans_accuracy =    [0.91, 0.68, 0.55, 0.50, 0.37]
lda_accuracy =       [0.89, 0.63, 0.50, 0.44, 0.32]
dbscan_accuracy =    [0.84, 0.60, 0.48, 0.40, 0.30]

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.plot(techniques, kmeans_accuracy, marker='o', label='K-Means')
plt.plot(techniques, lda_accuracy, marker='o', label='LDA')
plt.plot(techniques, dbscan_accuracy, marker='o', label='DBSCAN')

# Labels and Title
plt.title('Model Accuracy Degradation under DARD Manipulation Techniques')
plt.xlabel('DARD Technique')
plt.ylabel('Model Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
