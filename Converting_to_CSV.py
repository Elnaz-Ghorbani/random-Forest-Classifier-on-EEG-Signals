import os
import glob
import mne
import numpy as np
import pandas as pd

# Set the directory paths
data_dir = r'G:\University\Project_Intermship\An EEG Recordings Dataset for Mental Stress Detection\An EEG Recordings Dataset for Mental Stress Detection\clean data\Trier Mental Challenge Test (TMCT)'
output_dir = r'G:\University\Project_Intermship\An EEG Recordings Dataset for Mental Stress Detection\An EEG Recordings Dataset for Mental Stress Detection\clean data\Trier Mental Challenge Test (TMCT)_csv'
os.makedirs(output_dir, exist_ok=True)

# Get a list of all .set files in the directory
eeg_files = glob.glob(os.path.join(data_dir, '*.set'))

# Function to preprocess a single file and save time series data
def process_eeg_file(file_path):
    # Load the EEG data
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # Filter the data (0.5-50 Hz)
    raw.filter(l_freq=0.5, h_freq=50)
    
    # Remove powerline noise
    raw.notch_filter(freqs=50)
    
    # Convert the raw data to a numpy array
    data = raw.get_data()  # Shape: (n_channels, n_times)
    
    # Save the time series data to a CSV file
    file_name = os.path.basename(file_path).replace('.set', '.csv')
    csv_path = os.path.join(output_dir, file_name)
    df = pd.DataFrame(data.T)  # Transpose so that time points are rows
    df.to_csv(csv_path, index=False, header=False)  # Save without headers
    print(f"Saved processed data to {csv_path}")

# Process each file
for file_path in eeg_files:
    print(f"Processing file: {file_path}")
    process_eeg_file(file_path)

print("Data processing complete.")
