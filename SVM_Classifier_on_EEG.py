import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import random

# Define the paths to the folders containing the EEG data
folders = {
    'complex_math': "",
    'horror': "",
    'mental_test': ""
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
            array = df.values
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

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% of variance
data_reduced = pca.fit_transform(data_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_reduced, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Use the best parameters for the classifier
best_classifier = grid_search.best_estimator_

# Train the model
best_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Example Usage for Testing the Model
test_model_random_combination(
    folders, best_classifier, label_encoder, max_length, num_samples=10
)
test_model_multiple_repeats(
    folders, best_classifier, label_encoder, max_length, num_samples=5, num_repeats=10
)
