# Random Forest Classifier on EEG Signals

This repository contains scripts to convert EEG data files to CSV format, preprocess the data, and build a machine learning model using Random Forest to detect stressors in real-time.
## Repository Structure

- ### Converting_to_CSV.py
    This script converts EEG data from various formats (e.g., .edf, .set, .fdt) into CSV files, making them easier to process and analyze with machine learning models.

- ### Preprocess.py
    This script performs preprocessing on the converted EEG data. It includes:
  - Filtering the signals (e.g., removing noise, applying bandpass filters)
  - Normalizing the data
  - Reducing dimensionality (if applicable)
  - Splitting the data into training and testing sets

- ### Random_Forest_Classifier_on_EEG.py
    This script implements a Random Forest classifier to detect stressors based on EEG signals. It uses the preprocessed data to train the model and tests it on unseen data. The output is a classification that identifies whether the subject is under stress based on real-time EEG signals.

## Workflow

1. ### Convert EEG to CSV
    Use Converting_to_CSV.py to convert your EEG data into CSV format.

2. ### Preprocess Data
    Run Preprocess.py to clean and prepare the CSV data for machine learning. This step includes all necessary signal processing techniques to ensure quality input for the model.

3. ### Train and Test Classifier
    Use Random_Forest_Classifier_on_EEG.py to build a Random Forest model for real-time stress detection. This script handles the model training, testing, and evaluation of the classification results.
