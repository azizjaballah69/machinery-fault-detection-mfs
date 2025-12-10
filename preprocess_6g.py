import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal

# Load combined data
print("Loading combined data...")
data = pd.read_csv(r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\combined_6g_data.csv')

# Remove the 'source' column if it exists (it's just metadata)
if 'source' in data.columns:
    data = data.drop('source', axis=1)

print(f"Data loaded: {data.shape[0]} rows × {data.shape[1]} columns")

# Segmentation function
def segment_signal(signal_data, segment_size=1024, overlap=0.5):
    """Divide signal into overlapping segments"""
    step = int(segment_size * (1 - overlap))
    segments = []
    for start in range(0, len(signal_data) - segment_size, step):
        segments.append(signal_data[start:start + segment_size].values)
    return np.array(segments)

# Extract features from each segment
def extract_features(segment):
    """Extract statistical features from a segment"""
    return np.array([
        np.mean(segment),           # Mean
        np.std(segment),            # Standard deviation
        np.max(segment),            # Maximum
        np.min(segment),            # Minimum
        np.sqrt(np.mean(segment**2))  # RMS (Root Mean Square)
    ])

print("\nSegmenting signals...")
# Process each column (sensor)
features_list = []
for col in data.columns:
    segments = segment_signal(data[col], segment_size=1024, overlap=0.5)
    col_features = np.array([extract_features(seg) for seg in segments])
    features_list.append(col_features)
    print(f"Column {col}: {col_features.shape[0]} segments created")

# Combine all features
print("\nCombining features...")
X = np.hstack(features_list)
print(f"Combined feature matrix shape: {X.shape}")

# Normalize features using StandardScaler
print("\nNormalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create DataFrame for easier handling
feature_names = []
for i in range(data.shape[1]):
    feature_names.extend([f'col{i}_mean', f'col{i}_std', f'col{i}_max', f'col{i}_min', f'col{i}_rms'])

features_df = pd.DataFrame(X_scaled, columns=feature_names)

# Save processed features
output_features_file = r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g\processed_features.csv'
features_df.to_csv(output_features_file, index=False)
print(f"\nProcessed features saved to: {output_features_file}")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total samples: {X_scaled.shape[0]}")
print(f"Total features: {X_scaled.shape[1]}")
print(f"\nFeature statistics (normalized):")
print(features_df.describe())

print("\nFirst 10 feature samples:")
print(features_df.head(10))

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Visualize the features using matplotlib or seaborn")
print("2. Apply clustering (KMeans) or anomaly detection (Isolation Forest)")
print("3. Train a CNN model similar to your previous work")
print("4. If you have labels, perform supervised classification")
print("="*60)