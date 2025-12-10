import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load processed features
print("Loading processed features...")
features = pd.read_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\processed_features.csv')

print(f"Data shape: {features.shape}")

# Apply Isolation Forest for anomaly detection
print("\nApplying Isolation Forest for anomaly detection...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # 10% anomalies
anomaly_labels = iso_forest.fit_predict(features)
anomaly_scores = iso_forest.score_samples(features)

# Add results to dataframe
features['anomaly'] = anomaly_labels
features['anomaly_score'] = anomaly_scores

# Count anomalies
n_anomalies = (anomaly_labels == -1).sum()
print(f"Total samples: {len(features)}")
print(f"Anomalies detected: {n_anomalies} ({100*n_anomalies/len(features):.2f}%)")
print(f"Normal samples: {len(features) - n_anomalies}")

# Save results
output_file = r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\anomaly_detection_results.csv'
features.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Anomaly scores distribution
axes[0].hist(anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=iso_forest.offset_, color='r', linestyle='--', linewidth=2, label='Threshold')
axes[0].set_xlabel('Anomaly Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Anomaly Scores Distribution')
axes[0].legend()

# Plot 2: Normal vs Anomalies
normal_count = (anomaly_labels == 1).sum()
anomaly_count = (anomaly_labels == -1).sum()
axes[1].bar(['Normal', 'Anomalies'], [normal_count, anomaly_count], color=['green', 'red'], alpha=0.7)
axes[1].set_ylabel('Count')
axes[1].set_title('Normal vs Anomalous Samples')
for i, v in enumerate([normal_count, anomaly_count]):
    axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\05_anomaly_detection.png')
print("✓ Saved: 05_anomaly_detection.png")

# Print top 10 anomalies
print("\n" + "="*60)
print("TOP 10 MOST ANOMALOUS SAMPLES:")
print("="*60)
top_anomalies = features.nsmallest(10, 'anomaly_score')[['anomaly', 'anomaly_score']]
print(top_anomalies)

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Review anomalous samples to understand fault patterns")
print("2. Use these as labels for supervised model training")
print("3. Or adjust contamination parameter (0.05 to 0.15) to change sensitivity")
print("="*60)