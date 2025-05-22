import matplotlib.pyplot as plt

# Final values from your training graphs:
models = ['2D CNN', 'R(2+1)D']
val_accuracies = [68, 98]   # in %
val_losses = [1.0, 0.1]     # final val losses

# Create figure
plt.figure(figsize=(12, 5))

# ---- Accuracy Comparison ----
plt.subplot(1, 2, 1)
plt.bar(models, val_accuracies, color=['skyblue', 'lightgreen'])
plt.title('Validation Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
for i, v in enumerate(val_accuracies):
    plt.text(i, v + 2, f'{v}%', ha='center', fontsize=10)

# ---- Loss Comparison ----
plt.subplot(1, 2, 2)
plt.bar(models, val_losses, color=['orange', 'lightcoral'])
plt.title('Validation Loss Comparison')
plt.ylabel('Loss')
plt.ylim(0, 1.2)    
for i, v in enumerate(val_losses):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)

# Save & Show
plt.suptitle('Model Comparison: 2D CNN vs R(2+1)D', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("model_comparison_graphs.png")
plt.show()
