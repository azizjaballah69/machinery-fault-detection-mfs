import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load processed features
print("Loading processed features...")
features = pd.read_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\processed_features.csv')

print(f"Data shape: {features.shape}\n")

# 1. Statistical Summary
print("="*60)
print("STATISTICAL SUMMARY OF FEATURES")
print("="*60)
summary_stats = features.describe()
print(summary_stats)

# 2. Skewness and Kurtosis
print("\n" + "="*60)
print("DISTRIBUTION CHARACTERISTICS")
print("="*60)
skewness = features.skew()
kurtosis = features.kurtosis()
print(f"Average Skewness: {skewness.mean():.4f}")
print(f"Average Kurtosis: {kurtosis.mean():.4f}")

# 3. Feature importance based on variance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Variance)")
print("="*60)
feature_variance = features.var().sort_values(ascending=False)
print("Top 10 features by variance:")
print(feature_variance.head(10))

# 4. Correlation Analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
correlation_matrix = features.corr()
print("Features with highest mutual correlation (|r| > 0.7):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    corr_df = pd.DataFrame(high_corr_pairs)
    print(corr_df.to_string(index=False))
else:
    print("No feature pairs with |correlation| > 0.7")

# 5. Visualizations
fig = plt.figure(figsize=(16, 12))

# Heatmap of all correlations
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, ax=ax1, square=True)
ax1.set_title('Full Correlation Matrix Heatmap')

# Variance plot
ax2 = plt.subplot(2, 2, 2)
feature_variance.head(15).plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('Top 15 Features by Variance')
ax2.set_ylabel('Variance')
ax2.tick_params(axis='x', rotation=45)

# Distribution of mean values
ax3 = plt.subplot(2, 2, 3)
mean_features = [col for col in features.columns if '_mean' in col]
features[mean_features].boxplot(ax=ax3)
ax3.set_title('Distribution of Mean Features Across Sensors')
ax3.set_ylabel('Normalized Values')

# Distribution of RMS values
ax4 = plt.subplot(2, 2, 4)
rms_features = [col for col in features.columns if '_rms' in col]
features[rms_features].boxplot(ax=ax4)
ax4.set_title('Distribution of RMS Features Across Sensors')
ax4.set_ylabel('Normalized Values')

plt.tight_layout()
plt.savefig(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\06_statistical_analysis.png', dpi=100)
print("\n✓ Saved: 06_statistical_analysis.png")

# Save summary to CSV
summary_stats.to_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\feature_statistics.csv')
print("✓ Saved: feature_statistics.csv")

print("\n" + "="*60)
print("STATISTICAL ANALYSIS COMPLETE")
print("="*60)
print("Files generated:")
print("  1. 06_statistical_analysis.png (visualizations)")
print("  2. feature_statistics.csv (summary statistics)")
print("="*60)