"""
Banknote Authentication Using Self-Organizing Map (SOM)

Problem Statement:
------------------
The goal of this project is to classify banknotes as authentic or fraudulent using the Self-Organizing Map (SOM) algorithm.
The SOM will help visualize and cluster banknotes based on their features, identifying patterns that distinguish genuine from fake banknotes.


Dependencies:
-------------
- numpy
- pandas
- matplotlib
- scikit-learn
- MiniSom

Instructions:
-------------
1. Ensure that 'data_banknote_authentication.txt' is in the specified path.
2. Install the required libraries using:
    pip install numpy pandas matplotlib scikit-learn minisom
3. Run the script using:
    python som_banknote_authentication.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_dataset(file_path):
    """
    Load the Banknote Authentication dataset from a text file.

    Parameters:
    - file_path: str, path to the dataset file.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, target vector.
    """
    try:
        # The dataset does not have headers
        data = pd.read_csv(file_path, header=None)
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target
        print("Dataset loaded successfully.")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        return X, y
    except FileNotFoundError:
        print(f"File {file_path} not found. Please ensure the dataset is in the correct directory.")
        exit()

def preprocess_data(X):
    """
    Normalize the feature matrix using Min-Max scaling.

    Parameters:
    - X: np.ndarray, feature matrix.

    Returns:
    - X_scaled: np.ndarray, normalized feature matrix.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data normalization completed.")
    return X_scaled

def train_som(X, som_x=10, som_y=10, sigma=1.0, learning_rate=0.5, num_iterations=100):
    """
    Initialize and train the Self-Organizing Map (SOM).

    Parameters:
    - X: np.ndarray, normalized feature matrix.
    - som_x: int, width of the SOM grid.
    - som_y: int, height of the SOM grid.
    - sigma: float, radius of the different neighborhoods.
    - learning_rate: float, initial learning rate.
    - num_iterations: int, number of iterations for training.

    Returns:
    - som: MiniSom object, trained SOM.
    """
    som = MiniSom(x=som_x, y=som_y, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(X)
    print("SOM initialized. Starting training...")
    som.train_random(X, num_iteration=num_iterations)
    print("SOM training completed.")
    return som

def visualize_som(som, X, y):
    """
    Visualize the trained SOM with clusters.

    Parameters:
    - som: MiniSom object, trained SOM.
    - X: np.ndarray, normalized feature matrix.
    - y: np.ndarray, target vector.
    """
    plt.figure(figsize=(10, 10))
    plt.title('SOM - Banknote Authentication')

    # Plot the distance map as background
    plt.pcolor(som.distance_map().T, cmap='coolwarm')  # U-Matrix
    plt.colorbar(label='Distance')

    # Define markers and colors for classes
    markers = ['o', 's']
    colors = ['r', 'g']

    # Plot each sample on the SOM
    for idx, x in enumerate(X):
        w = som.winner(x)  # Get the winning node
        plt.plot(w[0] + 0.5, w[1] + 0.5, markers[y[idx]], markerfacecolor='None',
                 markeredgecolor=colors[y[idx]], markersize=12, markeredgewidth=2)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fake',
               markerfacecolor='None', markeredgecolor='r', markersize=10, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Genuine',
               markerfacecolor='None', markeredgecolor='g', markersize=10, markeredgewidth=2)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()

def map_classes_to_som(som, X, y):
    """
    Map each class to its winning node in the SOM.

    Parameters:
    - som: MiniSom object, trained SOM.
    - X: np.ndarray, normalized feature matrix.
    - y: np.ndarray, target vector.

    Returns:
    - class_map: dict, mapping of node positions to class labels.
    """
    class_map = {}
    for idx, x in enumerate(X):
        w = som.winner(x)
        if w not in class_map:
            class_map[w] = []
        class_map[w].append(y[idx])
    return class_map

def evaluate_som(class_map):
    """
    Evaluate the SOM by comparing cluster assignments with actual labels.

    Parameters:
    - class_map: dict, mapping of node positions to class labels.
    """
    # For simplicity, assign the majority class in each node
    y_pred = []
    y_true = []
    for node, labels in class_map.items():
        if labels:
            majority_label = max(set(labels), key=labels.count)
            y_pred.extend([majority_label] * len(labels))
            y_true.extend(labels)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

def main():
    """
    Main function to execute the SOM-based Banknote Authentication.
    """
    # Define file path (update to your specific path)
    dataset_path = os.path.join('C:\\Users\\User\\Downloads\\banknote+authentication', 'data_banknote_authentication.txt')

    # Load dataset
    X, y = load_dataset(dataset_path)

    # Preprocess data
    X_scaled = preprocess_data(X)

    # Train SOM
    som = train_som(X_scaled, som_x=10, som_y=10, sigma=1.0, learning_rate=0.5, num_iterations=100)

    # Visualize SOM
    visualize_som(som, X_scaled, y)

    # Map classes to SOM
    class_map = map_classes_to_som(som, X_scaled, y)

    # Evaluate SOM
    evaluate_som(class_map)

if __name__ == "__main__":
    main()
