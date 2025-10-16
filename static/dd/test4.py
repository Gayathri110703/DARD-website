import matplotlib.pyplot as plt
import seaborn as sns

# Simulated data
epochs = list(range(1, 21))

# Example accuracy values (you can plug in real ones)
train_accuracy = [0.65, 0.70, 0.74, 0.77, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89,
                  0.90, 0.91, 0.915, 0.918, 0.92, 0.922, 0.923, 0.925, 0.926, 0.927]

val_accuracy =   [0.63, 0.68, 0.73, 0.75, 0.78, 0.81, 0.83, 0.84, 0.85, 0.86,
                  0.87, 0.88, 0.885, 0.89, 0.892, 0.893, 0.895, 0.896, 0.897, 0.898]

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_accuracy, marker='o', label='Train Accuracy', color='green')
plt.plot(epochs, val_accuracy, marker='o', label='Validation Accuracy', color='blue')

# Labels and styling
plt.title('VAE Training vs. Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
