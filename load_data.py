import os
import pandas as pd

# Path to your data folder
folder = r'C:\Users\azizj\Downloads\imbalance (1)\imbalance\6g'

# All CSV filenames
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

# Load all CSV files
data_dict = {}
for filename in csv_files:
    filepath = os.path.join(folder, filename)
    data_dict[filename] = pd.read_csv(filepath, header=None)

print(f"Loaded {len(data_dict)} files successfully")

# Combine all files
all_data = []
for filename, df in data_dict.items():
    df['source'] = filename
    all_data.append(df)

combined_data = pd.concat(all_data, ignore_index=True)

print(f"Combined: {combined_data.shape[0]} rows × {combined_data.shape[1]} columns")
print("\nFirst 5 rows:")
print(combined_data.head())

# Save combined data
output_file = os.path.join(folder, 'combined_6g_data.csv')
combined_data.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")