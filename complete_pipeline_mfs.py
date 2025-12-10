"""
================================================================================
COMPLETE BEARING FAULT DETECTION PIPELINE - MFS DATASET
================================================================================
This script performs the complete workflow using the Machinery Fault Simulator
(MFS) dataset from UFRJ with imbalance faults:
https://www02.smt.ufrj.br/~offshore/mfs/page_01.html

Bearing fault types by filename pattern:
- Files with values < 30: HEALTHY bearings
- Files with values 30-40: LIGHT imbalance fault
- Files with values 40-60: MODERATE imbalance fault
- Files with values > 60: SEVERE imbalance fault

Steps:
1. Load all CSV files with imbalance labels
2. Combine and preprocess data
3. Extract features and normalize
4. Detect anomalies
5. Statistical analysis
6. Train machine learning model
7. Generate visualizations and reports
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FOLDER = r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g'
OUTPUT_FOLDER = DATA_FOLDER
WINDOW_SIZE = 1024
OVERLAP = 0.5
ANOMALY_CONTAMINATION = 0.1
RANDOM_STATE = 42

# ============================================================================
# SETUP & LOGGING
# ============================================================================
def setup_logging():
    """Create a log file for tracking progress"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_FOLDER, f'pipeline_log_{timestamp}.txt')
    return log_file

def log_message(message, log_file=None):
    """Print and save message to log"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

# ============================================================================
# FAULT LABELING FUNCTION
# ============================================================================
def get_fault_label(filename):
    """
    Extract fault severity from filename based on MFS dataset convention
    
    Filename format: XX.YYYY.csv where XX.YYYY is the load/imbalance value
    - Values < 30: HEALTHY (label 0)
    - Values 30-40: LIGHT imbalance (label 1)
    - Values 40-60: MODERATE imbalance (label 2)
    - Values > 60: SEVERE imbalance (label 3)
    """
    try:
        # Extract numeric part from filename (e.g., "60.2112" from "60.2112.csv")
        value_str = filename.replace('.csv', '')
        value = float(value_str)
        
        if value < 30:
            return 0, 'HEALTHY'
        elif value < 40:
            return 1, 'LIGHT_IMBALANCE'
        elif value < 60:
            return 2, 'MODERATE_IMBALANCE'
        else:
            return 3, 'SEVERE_IMBALANCE'
    except:
        return -1, 'UNKNOWN'

# ============================================================================
# STEP 1: LOAD AND COMBINE DATA WITH FAULT LABELS
# ============================================================================
def load_and_combine_data(log_file=None):
    """Load all CSV files and combine them with fault labels"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 1: LOADING AND COMBINING DATA (MFS DATASET - IMBALANCE FAULTS)", log_file)
    log_message("="*70, log_file)
    
    csv_files = [
        '60.2112.csv', '30.5152.csv', '62.0544.csv', '31.5392.csv',
        '58.1632.csv', '28.8768.csv', '58.7776.csv', '29.4912.csv',
        '59.392.csv', '26.8288.csv', '54.4768.csv', '27.648.csv',
        '55.296.csv', '24.576.csv', '56.1152.csv', '25.3952.csv',
        '52.224.csv', '18.432.csv', '53.6576.csv', '19.6608.csv',
        '50.5856.csv', '20.2752.csv', '51.8144.csv', '21.7088.csv',
        '46.6944.csv', '22.7328.csv', '47.3088.csv', '23.552.csv',
        '48.128.csv', '15.36.csv', '49.152.csv', '16.1792.csv',
        '44.4416.csv', '17.408.csv', '45.4656.csv', '13.9264.csv',
        '42.5984.csv', '14.336.csv', '43.4176.csv', '40.7552.csv',
        '41.3696.csv', '37.6832.csv', '38.5024.csv', '39.3216.csv',
        '36.4544.csv', '34.816.csv', '35.6352.csv', '32.9728.csv',
        '33.9968.csv'
    ]
    
    log_message(f"Loading {len(csv_files)} CSV files from MFS dataset...", log_file)
    log_message("\nFault Classification Scheme:", log_file)
    log_message("  Label 0: HEALTHY (value < 30)", log_file)
    log_message("  Label 1: LIGHT IMBALANCE (30 ≤ value < 40)", log_file)
    log_message("  Label 2: MODERATE IMBALANCE (40 ≤ value < 60)", log_file)
    log_message("  Label 3: SEVERE IMBALANCE (value ≥ 60)", log_file)
    
    data_dict = {}
    fault_labels = {}
    
    for filename in csv_files:
        filepath = os.path.join(DATA_FOLDER, filename)
        try:
            data_dict[filename] = pd.read_csv(filepath, header=None)
            fault_id, fault_name = get_fault_label(filename)
            fault_labels[filename] = (fault_id, fault_name)
        except Exception as e:
            log_message(f"Error loading {filename}: {e}", log_file)
    
    log_message(f"\n✓ Loaded {len(data_dict)} files successfully", log_file)
    
    # Display fault distribution
    log_message("\nFault Distribution:", log_file)
    fault_counts = {}
    for filename, (fault_id, fault_name) in fault_labels.items():
        if fault_name not in fault_counts:
            fault_counts[fault_name] = 0
        fault_counts[fault_name] += 1
    
    for fault_name, count in sorted(fault_counts.items()):
        log_message(f"  {fault_name}: {count} files", log_file)
    
    # Combine all data with labels
    all_data = []
    for filename, df in data_dict.items():
        fault_id, fault_name = fault_labels[filename]
        df['source'] = filename
        df['fault_label'] = fault_id
        df['fault_name'] = fault_name
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    log_message(f"\n✓ Combined: {combined_data.shape[0]} rows × {combined_data.shape[1]} columns", log_file)
    
    # Save combined data
    combined_file = os.path.join(OUTPUT_FOLDER, 'combined_6g_data_with_labels.csv')
    combined_data.to_csv(combined_file, index=False)
    log_message(f"✓ Saved: {combined_file}", log_file)
    
    return combined_data

# ============================================================================
# STEP 2: FEATURE EXTRACTION & PREPROCESSING
# ============================================================================
def extract_features(data, log_file=None):
    """Extract statistical features from signal windows"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 2: FEATURE EXTRACTION & PREPROCESSING", log_file)
    log_message("="*70, log_file)
    
    # Extract fault labels before removing columns
    fault_labels_full = data['fault_label'].values
    fault_names_full = data['fault_name'].values
    
    # Remove label columns
    data_values = data.iloc[:, :-3].values  # Remove source, fault_label, fault_name
    n_sensors = data_values.shape[1]
    
    log_message(f"Extracting features from {n_sensors} sensors...", log_file)
    log_message(f"Window size: {WINDOW_SIZE}, Overlap: {OVERLAP*100}%", log_file)
    
    step = int(WINDOW_SIZE * (1 - OVERLAP))
    all_features = []
    segment_fault_labels = []
    segment_fault_names = []
    
    for col_idx in range(n_sensors):
        signal_data = data_values[:, col_idx]
        windows = []
        
        for start in range(0, len(signal_data) - WINDOW_SIZE + 1, step):
            window = signal_data[start:start + WINDOW_SIZE]
            windows.append(window)
        
        if col_idx == 0:  # Only log for first sensor
            log_message(f"  Sensor {col_idx}: {len(windows)} segments created", log_file)
            # Store fault labels for segments (one label per segment from the first row of each window)
            for start in range(0, len(signal_data) - WINDOW_SIZE + 1, step):
                window_idx = start // step
                if window_idx < len(fault_labels_full):
                    segment_fault_labels.append(fault_labels_full[start])
                    segment_fault_names.append(fault_names_full[start])
        
        # Extract features for each window
        features_list = []
        for window in windows:
            features_list.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'max': np.max(window),
                'min': np.min(window),
                'rms': np.sqrt(np.mean(window**2))
            })
        
        for feat_name, _ in features_list[0].items():
            all_features.append([f[feat_name] for f in features_list])
    
    # Transpose to get (samples, features)
    feature_matrix = np.array(all_features).T
    log_message(f"✓ Feature matrix shape: {feature_matrix.shape}", log_file)
    
    # Normalize features
    log_message("Normalizing features...", log_file)
    scaler = StandardScaler()
    feature_matrix_normalized = scaler.fit_transform(feature_matrix)
    
    # Create dataframe with feature names and fault labels
    feature_names = [f'col{i%n_sensors}_{["mean","std","max","min","rms"][i//n_sensors]}' 
                     for i in range(feature_matrix_normalized.shape[1])]
    features_df = pd.DataFrame(feature_matrix_normalized, columns=feature_names)
    features_df['fault_label'] = segment_fault_labels
    features_df['fault_name'] = segment_fault_names
    
    log_message(f"\nSegment Fault Distribution:", log_file)
    fault_dist = features_df['fault_name'].value_counts().sort_index()
    for fault_name, count in fault_dist.items():
        log_message(f"  {fault_name}: {count} segments ({100*count/len(features_df):.1f}%)", log_file)
    
    # Save processed features
    features_file = os.path.join(OUTPUT_FOLDER, 'processed_features_with_labels.csv')
    features_df.to_csv(features_file, index=False)
    log_message(f"\n✓ Saved: {features_file}", log_file)
    
    return features_df, scaler

# ============================================================================
# STEP 3: ANOMALY DETECTION
# ============================================================================
def detect_anomalies(features_df, log_file=None):
    """Detect anomalies using Isolation Forest"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 3: ANOMALY DETECTION", log_file)
    log_message("="*70, log_file)
    
    # Remove fault columns for anomaly detection
    feature_cols = [col for col in features_df.columns if col not in ['fault_label', 'fault_name']]
    
    log_message(f"Applying Isolation Forest (contamination={ANOMALY_CONTAMINATION})...", log_file)
    
    iso_forest = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=RANDOM_STATE)
    anomaly_labels = iso_forest.fit_predict(features_df[feature_cols])
    anomaly_scores = iso_forest.score_samples(features_df[feature_cols])
    
    features_df['is_anomaly'] = anomaly_labels
    features_df['anomaly_score'] = anomaly_scores
    
    n_anomalies = (anomaly_labels == -1).sum()
    log_message(f"\n✓ Total samples: {len(features_df)}", log_file)
    log_message(f"✓ Anomalies detected: {n_anomalies} ({100*n_anomalies/len(features_df):.2f}%)", log_file)
    log_message(f"✓ Normal samples: {len(features_df) - n_anomalies}", log_file)
    
    # Show anomalies by fault type
    log_message(f"\nAnomalies by Fault Type:", log_file)
    anomaly_df = features_df[features_df['is_anomaly'] == -1]
    for fault_name in sorted(features_df['fault_name'].unique()):
        fault_anomalies = len(anomaly_df[anomaly_df['fault_name'] == fault_name])
        fault_total = len(features_df[features_df['fault_name'] == fault_name])
        pct = 100 * fault_anomalies / fault_total if fault_total > 0 else 0
        log_message(f"  {fault_name}: {fault_anomalies}/{fault_total} ({pct:.1f}%)", log_file)
    
    # Save anomaly results
    anomaly_file = os.path.join(OUTPUT_FOLDER, 'anomaly_detection_results_with_labels.csv')
    features_df.to_csv(anomaly_file, index=False)
    log_message(f"\n✓ Saved: {anomaly_file}", log_file)
    
    return features_df, iso_forest

# ============================================================================
# STEP 4: STATISTICAL ANALYSIS
# ============================================================================
def statistical_analysis(features_df, log_file=None):
    """Perform statistical analysis on features"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 4: STATISTICAL ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    # Remove label/anomaly columns for analysis
    feature_cols = [col for col in features_df.columns if col not in ['fault_label', 'fault_name', 'is_anomaly', 'anomaly_score']]
    features_analysis = features_df[feature_cols]
    
    log_message("Calculating statistics...", log_file)
    summary_stats = features_analysis.describe()
    
    skewness = features_analysis.skew()
    kurtosis = features_analysis.kurtosis()
    
    log_message(f"Average Skewness: {skewness.mean():.4f}", log_file)
    log_message(f"Average Kurtosis: {kurtosis.mean():.4f}", log_file)
    
    # Save statistics
    stats_file = os.path.join(OUTPUT_FOLDER, 'feature_statistics.csv')
    summary_stats.to_csv(stats_file)
    log_message(f"✓ Saved: {stats_file}", log_file)
    
    return summary_stats

# ============================================================================
# STEP 5: TRAIN MACHINE LEARNING MODEL WITH TRUE FAULT LABELS
# ============================================================================
def train_ml_model(features_df, log_file=None):
    """Train Random Forest classifier using true fault labels"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 5: MACHINE LEARNING MODEL TRAINING (SUPERVISED)", log_file)
    log_message("="*70, log_file)
    
    # Remove anomaly columns
    feature_cols = [col for col in features_df.columns if col not in ['fault_label', 'fault_name', 'is_anomaly', 'anomaly_score']]
    X = features_df[feature_cols].values
    y = features_df['fault_label'].values  # True fault labels
    
    log_message(f"\nUsing TRUE fault labels from MFS dataset:", log_file)
    unique_labels = np.unique(y)
    for label in sorted(unique_labels):
        count = np.sum(y == label)
        fault_name = features_df[features_df['fault_label'] == label]['fault_name'].iloc[0]
        log_message(f"  Label {label} ({fault_name}): {count} samples ({100*count/len(y):.1f}%)", log_file)
    
    # Split data
    log_message("\nSplitting data (70% train, 15% val, 15% test)...", log_file)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, 
                                                         random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
                                                     random_state=RANDOM_STATE, stratify=y_temp)
    
    log_message(f"  Train: {X_train.shape[0]} samples", log_file)
    log_message(f"  Val: {X_val.shape[0]} samples", log_file)
    log_message(f"  Test: {X_test.shape[0]} samples", log_file)
    
    # Train model
    log_message("\nTraining Random Forest Classifier...", log_file)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                      random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    log_message(f"\n✓ Train Accuracy: {train_acc*100:.2f}%", log_file)
    log_message(f"✓ Val Accuracy: {val_acc*100:.2f}%", log_file)
    log_message(f"✓ Test Accuracy: {test_acc*100:.2f}%", log_file)
    
    # Classification report
    log_message(f"\nDetailed Classification Report (Test Set):", log_file)
    log_message(classification_report(y_test, y_test_pred), log_file)
    
    # Save model
    model_file = os.path.join(OUTPUT_FOLDER, 'random_forest_model_mfs.pkl')
    joblib.dump(rf_model, model_file)
    log_message(f"✓ Model saved: {model_file}", log_file)
    
    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, 'model_predictions_mfs.csv')
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_test_pred,
        'correct': y_test == y_test_pred
    })
    results_df.to_csv(predictions_file, index=False)
    log_message(f"✓ Predictions saved: {predictions_file}", log_file)
    
    return rf_model, (X_train, X_val, X_test), (y_train, y_val, y_test), (train_acc, val_acc, test_acc)

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
def create_visualizations(features_df, rf_model, accuracies, log_file=None):
    """Create comprehensive visualizations"""
    log_message("\n" + "="*70, log_file)
    log_message("STEP 6: CREATING VISUALIZATIONS", log_file)
    log_message("="*70, log_file)
    
    feature_cols = [col for col in features_df.columns if col not in ['fault_label', 'fault_name', 'is_anomaly', 'anomaly_score']]
    
    # Plot 1: Feature Distribution by Fault Type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Distributions by Fault Type (MFS Dataset - Imbalance Faults)', fontsize=14)
    
    fault_names = sorted(features_df['fault_name'].unique())
    colors = ['green', 'yellow', 'orange', 'red']
    
    for idx, fault_name in enumerate(fault_names):
        fault_data = features_df[features_df['fault_name'] == fault_name]['col0_mean']
        axes[idx // 2, idx % 2].hist(fault_data, bins=30, edgecolor='black', alpha=0.7, color=colors[idx])
        axes[idx // 2, idx % 2].set_title(fault_name)
        axes[idx // 2, idx % 2].set_xlabel('Mean Value')
        axes[idx // 2, idx % 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '01_fault_distributions.png'), dpi=100)
    plt.close()
    log_message("✓ Saved: 01_fault_distributions.png", log_file)
    
    # Plot 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = features_df[feature_cols].corr()
    sns.heatmap(correlation.iloc[:20, :20], cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap (First 20 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '02_correlation_heatmap.png'), dpi=100)
    plt.close()
    log_message("✓ Saved: 02_correlation_heatmap.png", log_file)
    
    # Plot 3: Anomaly Detection Results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(features_df['anomaly_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_title('Anomaly Scores Distribution')
    axes[0].grid(alpha=0.3)
    
    n_normal = (features_df['is_anomaly'] == 1).sum()
    n_anomaly = (features_df['is_anomaly'] == -1).sum()
    axes[1].bar(['Normal', 'Anomalies'], [n_normal, n_anomaly], color=['green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Sample Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '03_anomaly_analysis.png'), dpi=100)
    plt.close()
    log_message("✓ Saved: 03_anomaly_analysis.png", log_file)
    
    # Plot 4: Model Performance & Fault Classification
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Feature importance
    ax1 = fig.add_subplot(gs[0, :2])
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(15)
    ax1.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values)
    ax1.set_title('Top 15 Feature Importances')
    ax1.invert_yaxis()
    
    # Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    train_acc, val_acc, test_acc = accuracies
    datasets = ['Train', 'Val', 'Test']
    acc_values = [train_acc*100, val_acc*100, test_acc*100]
    ax2.bar(datasets, acc_values, color=['green', 'orange', 'blue'], alpha=0.7)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Accuracy by Dataset')
    ax2.set_ylim([0, 105])
    for i, v in enumerate(acc_values):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Fault distribution
    ax3 = fig.add_subplot(gs[1, 0])
    fault_dist = features_df['fault_name'].value_counts()
    colors_fault = ['green', 'yellow', 'orange', 'red']
    ax3.pie(fault_dist.values, labels=fault_dist.index, autopct='%1.1f%%', colors=colors_fault)
    ax3.set_title('Sample Distribution by Fault Type')
    
    # Summary stats
    ax4 = fig.add_subplot(gs[1, 1:])
    summary_text = f"""
    MFS Dataset - Bearing Imbalance Faults
    ════════════════════════════════════════════════════
    
    Training Summary:
    • Train Accuracy: {train_acc*100:.2f}%
    • Val Accuracy: {val_acc*100:.2f}%
    • Test Accuracy: {test_acc*100:.2f}%
    • Total Features: {len(feature_cols)}
    • Total Samples: {len(features_df)}
    
    Fault Types (4 Classes):
    • HEALTHY: {len(features_df[features_df['fault_label'] == 0])} segments
    • LIGHT IMBALANCE: {len(features_df[features_df['fault_label'] == 1])} segments
    • MODERATE IMBALANCE: {len(features_df[features_df['fault_label'] == 2])} segments
    • SEVERE IMBALANCE: {len(features_df[features_df['fault_label'] == 3])} segments
    
    Dataset Source:
    UFRJ - Machinery Fault Simulator (MFS)
    https://www02.smt.ufrj.br/~offshore/mfs/page_01.html
    """
    ax4.text(0.05, 0.95, summary_text, fontsize=10, transform=ax4.transAxes, 
             verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, '04_model_performance_mfs.png'), dpi=100)
    plt.close()
    log_message("✓ Saved: 04_model_performance_mfs.png", log_file)

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("BEARING FAULT DETECTION - COMPLETE MFS PIPELINE")
    print("Dataset: UFRJ Machinery Fault Simulator (Imbalance Faults)")
    print("="*70)
    
    # Setup logging
    log_file = setup_logging()
    log_message(f"Pipeline started at {datetime.now()}", log_file)
    log_message(f"Output folder: {OUTPUT_FOLDER}", log_file)
    log_message(f"Dataset: UFRJ MFS - https://www02.smt.ufrj.br/~offshore/mfs/page_01.html", log_file)
    
    try:
        # Step 1: Load and combine data with fault labels
        combined_data = load_and_combine_data(log_file)
        
        # Step 2: Extract features
        features_df, scaler = extract_features(combined_data, log_file)
        
        # Step 3: Anomaly detection
        features_df, iso_forest = detect_anomalies(features_df, log_file)
        
        # Step 4: Statistical analysis
        stats = statistical_analysis(features_df, log_file)
        
        # Step 5: Train ML model with TRUE fault labels
        rf_model, data_splits, labels_splits, accuracies = train_ml_model(features_df, log_file)
        
        # Step 6: Create visualizations
        create_visualizations(features_df, rf_model, accuracies, log_file)
        
        # Final summary
        log_message("\n" + "="*70, log_file)
        log_message("PIPELINE COMPLETED SUCCESSFULLY!", log_file)
        log_message("="*70, log_file)
        log_message(f"Completed at {datetime.now()}", log_file)
        log_message("\nOutput files:", log_file)
        log_message("  - combined_6g_data_with_labels.csv", log_file)
        log_message("  - processed_features_with_labels.csv", log_file)
        log_message("  - anomaly_detection_results_with_labels.csv", log_file)
        log_message("  - feature_statistics.csv", log_file)
        log_message("  - random_forest_model_mfs.pkl", log_file)
        log_message("  - model_predictions_mfs.csv", log_file)
        log_message("  - 01_fault_distributions.png", log_file)
        log_message("  - 02_correlation_heatmap.png", log_file)
        log_message("  - 03_anomaly_analysis.png", log_file)
        log_message("  - 04_model_performance_mfs.png", log_file)
        log_message(f"  - pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", log_file)
        log_message("="*70, log_file)
        
    except Exception as e:
        log_message(f"\n❌ ERROR: {str(e)}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)

if __name__ == "__main__":
    main()