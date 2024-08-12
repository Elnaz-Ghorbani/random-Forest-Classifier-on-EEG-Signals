import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define the paths to the folders containing the EEG data
folders = {
    'complex_math': r'G:\University\Project_Intermship\An EEG Recordings Dataset for Mental Stress Detection\An EEG Recordings Dataset for Mental Stress Detection\clean data\CSVs\Complex Mathematical Problem solving (CMPS)',
    'horror': r'G:\University\Project_Intermship\An EEG Recordings Dataset for Mental Stress Detection\An EEG Recordings Dataset for Mental Stress Detection\clean data\CSVs\Horrer Video Stimulation',
    'mental_test': r'G:\University\Project_Intermship\An EEG Recordings Dataset for Mental Stress Detection\An EEG Recordings Dataset for Mental Stress Detection\clean data\CSVs\Trier Mental Challenge Test (TMCT)'
}

# Find the maximum sequence length
max_length = 0
for folder in folders.values():
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            max_length = max(max_length, len(df))

# Preprocess the data
for label, folder in folders.items():
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            array = df.values
            
            # Padding to max_length
            if len(array) < max_length:
                padded_array = np.pad(array, ((0, max_length - len(array)), (0, 0)), mode='constant')
            else:
                padded_array = array[:max_length]
            
            # Normalize the data
            scaler = StandardScaler()
            padded_array_scaled = scaler.fit_transform(padded_array)
            
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=min(padded_array_scaled.shape[0], padded_array_scaled.shape[1]))
            pca_array = pca.fit_transform(padded_array_scaled)
            
            # Save the preprocessed data to overwrite the original file
            preprocessed_df = pd.DataFrame(pca_array)
            preprocessed_df.to_csv(filepath, index=False)
            
            print(f'Preprocessed and saved with PCA: {filepath}')

print('Data preprocessing with PCA complete.')
