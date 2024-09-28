import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

# Define the paths to the folders containing the EEG data
folders = {
    'complex_math':"",
    'horror': "",
    'mental_test': "",
    'color_word': ""
}

# Initialize lists to store the data and labels
data = []
labels = []

# Find the maximum sequence length
max_length = 0
for folder in folders.values():
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            max_length = max(max_length, len(df))

# Loop through each folder and read the CSV files
for label, folder in folders.items():
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            # Convert DataFrame to a NumPy array
            array = df.values
            # Pad the array to the maximum length with zeros
            if len(array) < max_length:
                padded_array = np.pad(array, ((0, max_length - len(array)), (0, 0)), mode='constant')
            else:
                padded_array = array
            data.append(padded_array.flatten())
            labels.append(label)

# Convert the data and labels into NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.3, random_state=42)

# Initialize and train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

def test_model_random_combination(folders, classifier, label_encoder, max_length, num_samples=1):
    # Initialize lists to store the real and predicted labels
    real_labels = []
    predicted_labels = []
    
    # Initialize dictionaries to track incorrect predictions for each condition
    incorrect_counts = {label: 0 for label in folders.keys()}
    total_counts = {label: 0 for label in folders.keys()}
    
    # Gather all selected samples into a list
    samples = []
    
    # Loop over each condition
    for label, folder in folders.items():
        filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Randomly select the specified number of CSV files
        selected_filenames = random.sample(filenames, num_samples)
        
        for random_filename in selected_filenames:
            filepath = os.path.join(folder, random_filename)
            samples.append((label, filepath))
    
    # Shuffle the samples list to randomize the order
    random.shuffle(samples)
    
    # Process each sample in the shuffled list
    for label, filepath in samples:
        df = pd.read_csv(filepath)
        array = df.values
        
        # Ensure the data has the same length as the training data
        if len(array) < max_length:
            padded_array = np.pad(array, ((0, max_length - len(array)), (0, 0)), mode='constant')
        else:
            padded_array = array[:max_length]
        
        # Flatten the data to match the model's expected input
        test_data = padded_array.flatten().reshape(1, -1)
        
        # Predict the condition using the trained model
        predicted_label_encoded = classifier.predict(test_data)
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
        
        # Store the real and predicted labels
        real_labels.append(label)
        predicted_labels.append(predicted_label[0])
        
        # Update counts for the current condition
        total_counts[label] += 1
        if label != predicted_label[0]:
            incorrect_counts[label] += 1
    
    # Calculate the overall incorrect rate
    incorrect_predictions = sum(incorrect_counts.values())
    incorrect_rate = incorrect_predictions / len(real_labels)
    
    # Print the real and predicted labels
    print(f'Real Labels: {real_labels}')
    print(f'Predicted Labels: {predicted_labels}')
    print(f'Overall Incorrect Rate: {incorrect_rate:.2f}')
    print(f'Total Test Samples: {len(real_labels)}')
    
    # Print incorrect rates for each condition
    for label in folders.keys():
        condition_incorrect_rate = incorrect_counts[label] / total_counts[label] if total_counts[label] > 0 else 0
        print(f'Incorrect Rate for {label}: {condition_incorrect_rate:.2f}')

# Example usage: test with 22 samples from each condition
test_model_random_combination(folders, classifier, label_encoder, max_length, num_samples=10)

def test_model_multiple_repeats(folders, classifier, label_encoder, max_length, num_samples=1, num_repeats=10):
    # Initialize a dictionary to store incorrect rates for each condition across repeats
    condition_incorrect_rates = {label: [] for label in folders.keys()}
    overall_incorrect_rates = []
    
    # Repeat the process num_repeats times
    for _ in range(num_repeats):
        # Initialize dictionaries to track incorrect predictions for each condition
        incorrect_counts = {label: 0 for label in folders.keys()}
        total_counts = {label: 0 for label in folders.keys()}
        
        # Gather all selected samples into a list
        samples = []
        
        # Loop over each condition
        for label, folder in folders.items():
            filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
            
            # Randomly select the specified number of CSV files
            selected_filenames = random.sample(filenames, num_samples)
            
            for random_filename in selected_filenames:
                filepath = os.path.join(folder, random_filename)
                samples.append((label, filepath))
        
        # Shuffle the samples list to randomize the order
        random.shuffle(samples)
        
        # Process each sample in the shuffled list
        for label, filepath in samples:
            df = pd.read_csv(filepath)
            array = df.values
            
            # Ensure the data has the same length as the training data
            if len(array) < max_length:
                padded_array = np.pad(array, ((0, max_length - len(array)), (0, 0)), mode='constant')
            else:
                padded_array = array[:max_length]
            
            # Flatten the data to match the model's expected input
            test_data = padded_array.flatten().reshape(1, -1)
            
            # Predict the condition using the trained model
            predicted_label_encoded = classifier.predict(test_data)
            predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
            
            # Update counts for the current condition
            total_counts[label] += 1
            if label != predicted_label[0]:
                incorrect_counts[label] += 1
        
        # Calculate the incorrect rate for each condition in this repeat
        for label in folders.keys():
            condition_incorrect_rate = incorrect_counts[label] / total_counts[label] if total_counts[label] > 0 else 0
            condition_incorrect_rates[label].append(condition_incorrect_rate)
        
        # Calculate the overall incorrect rate for this repeat
        incorrect_predictions = sum(incorrect_counts.values())
        overall_incorrect_rate = incorrect_predictions / sum(total_counts.values())
        overall_incorrect_rates.append(overall_incorrect_rate)
    
    # Calculate and print the average incorrect rate for each condition across all repeats
    for label in folders.keys():
        average_incorrect_rate = sum(condition_incorrect_rates[label]) / num_repeats
        print(f'Average Incorrect Rate for {label} over {num_repeats} repeats: {average_incorrect_rate:.2f}')
    
    # Calculate and print the overall average incorrect rate across all conditions
    overall_average_incorrect_rate = sum(overall_incorrect_rates) / num_repeats
    print(f'Overall Average Incorrect Rate over {num_repeats} repeats: {overall_average_incorrect_rate:.2f}')

# Example usage: test with 22 samples from each condition, repeated 10 times
test_model_multiple_repeats(folders, classifier, label_encoder, max_length, num_samples=5, num_repeats=10)

