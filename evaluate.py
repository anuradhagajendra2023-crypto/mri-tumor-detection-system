import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================
# Paths
# ============================
model_path = "brain_tumor_model.keras"
test_path = "Testing"

# ============================
# Load Model
# ============================
model = load_model(model_path)

# ============================
# Load Test Data (no shuffle -> order matches labels)
# ============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())
print("Classes:", test_data.class_indices)

# ============================
# Predictions
# ============================
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# ============================
# Accuracy
# ============================
acc = accuracy_score(true_classes, predicted_classes)
print(f"\nOverall Test Accuracy: {acc*100:.2f}%\n")

# ============================
# Classification Report
# ============================
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("Classification Report:\n")
print(report)

# Save report to a text file
with open("classification_report.txt", "w") as f:
    f.write(f"Overall Test Accuracy: {acc*100:.2f}%\n\n")
    f.write(report)

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Brain Tumor Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\nSaved: classification_report.txt and confusion_matrix.png")