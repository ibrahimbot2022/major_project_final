import matplotlib.pyplot as plt

# Data
models = ['2D CNN', 'R(2+1)D']
val_accuracies = [68, 98]  # Final val accuracies (%)
val_losses = [1.0, 0.1]    # Final val losses

# Plotting
plt.figure(figsize=(12, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
plt.bar(models, val_accuracies, color=['skyblue', 'lightgreen'])
plt.title('Validation Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)

# Loss comparison
plt.subplot(1, 2, 2)
plt.bar(models, val_losses, color=['orange', 'lightcoral'])
plt.title('Validation Loss Comparison')
plt.ylabel('Loss')
plt.ylim(0, 1.2)

plt.tight_layout()
plt.savefig("comparison_graphs.png")
plt.show()
