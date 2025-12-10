import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading processed features...")
features = pd.read_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\processed_features.csv')
X = features.values

print(f"Feature matrix shape: {X.shape}")

# Create pseudo-labels using KMeans (3 bearing conditions)
print("\nCreating pseudo-labels using KMeans clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y = kmeans.fit_predict(X)

print(f"Cluster distribution:")
unique, counts = np.unique(y, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples ({100*count/len(y):.1f}%)")

# Split data
print("\nSplitting data (70% train, 15% val, 15% test)...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} samples")
print(f"Val: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Train Random Forest Model
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=1)
rf_model.fit(X_train, y_train)

# Predictions
print("\nMaking predictions...")
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

# Calculate accuracies
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nAccuracy Results:")
print(f"  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Val Accuracy: {val_acc*100:.2f}%")
print(f"  Test Accuracy: {test_acc*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Feature Importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': [f'col{i%8}_{["mean","std","max","min","rms"][i//8]}' for i in range(40)],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Visualizations
fig = plt.figure(figsize=(16, 10))

# 1. Feature Importance
ax1 = plt.subplot(2, 2, 1)
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features['importance'].values)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'].values)
ax1.set_xlabel('Importance')
ax1.set_title('Top 15 Feature Importances')
ax1.invert_yaxis()

# 2. Accuracy Comparison
ax2 = plt.subplot(2, 2, 2)
datasets = ['Train', 'Validation', 'Test']
accuracies = [train_acc*100, val_acc*100, test_acc*100]
colors = ['green', 'orange', 'blue']
ax2.bar(datasets, accuracies, color=colors, alpha=0.7)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Model Accuracy by Dataset')
ax2.set_ylim([0, 105])
for i, v in enumerate(accuracies):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 3. Confusion Matrix
ax3 = plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_title('Confusion Matrix (Test Set)')

# 4. Class Distribution
ax4 = plt.subplot(2, 2, 4)
unique, counts = np.unique(y, return_counts=True)
ax4.bar([f'Cluster {c}' for c in unique], counts, color=['red', 'green', 'blue'], alpha=0.7)
ax4.set_ylabel('Number of Samples')
ax4.set_title('Data Distribution by Cluster')

plt.tight_layout()
plt.savefig(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\07_random_forest_results.png', dpi=100)
print("\n✓ Saved: 07_random_forest_results.png")

# Save model
import joblib
joblib.dump(rf_model, r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\random_forest_model.pkl')
print("✓ Model saved: random_forest_model.pkl")

# Save predictions
results_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_test_pred,
    'prediction_correct': y_test == y_test_pred
})
results_df.to_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\random_forest_predictions.csv', index=False)
print("✓ Predictions saved: random_forest_predictions.csv")

print("\n" + "="*60)
print("RANDOM FOREST MODEL TRAINING COMPLETE")
print("="*60)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print("\nFiles generated:")
print("  1. random_forest_model.pkl (trained model)")
print("  2. random_forest_predictions.csv (test predictions)")
print("  3. 07_random_forest_results.png (visualizations)")
print("="*60)