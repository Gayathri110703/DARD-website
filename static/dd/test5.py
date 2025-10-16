import matplotlib.pyplot as plt

# Sample accuracy values (replace with your actual values)
training_accuracy = 0.95
validation_accuracy = 0.94

# Bar labels and corresponding accuracy values
labels = ['Training Accuracy', 'Validation Accuracy']
accuracies = [training_accuracy, validation_accuracy]

# Create bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, accuracies, color=['skyblue', 'salmon'])

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

# Graph details
plt.ylim(0, 1)
plt.title('VAE Training vs. Validation Accuracy')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/graph4.png')
plt.show()
