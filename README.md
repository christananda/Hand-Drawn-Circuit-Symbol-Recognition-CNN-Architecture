Hand-Drawn Circuit Symbol Recognition using Convolutional Neural Networks (CNN)
📌 Project Overview
This project develops a Convolutional Neural Network (CNN) designed to accurately classify hand-drawn electric circuit schematic components. The goal is to build a robust model capable of handling the inherent noise, variations, and inconsistencies present in hand-drawn symbols, making it ideal for digitizing hand-sketched circuit diagrams.

The model is trained on the Kaggle dataset: "Hand-drawn Electric circuit Schematic Components."

✨ Key Features
15-Class Classification: The model can distinguish between 15 different common circuit components (e.g., Resistor, Capacitor, AC Source, etc.).

Robustness via Data Augmentation: Uses advanced augmentation techniques (rotation, zoom, translation) to simulate real-world drawing variations, significantly improving generalization.

High Performance: Achieves [Placeholder: Insert Final Validation Accuracy Here] validation accuracy across 30 epochs.

Explainability (Grad-CAM): Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visually highlight the exact regions of the image the CNN uses to make its prediction, aiding in model interpretability.

🧠 Architecture Overview
The solution utilizes a custom deep CNN architecture, CircuitSymbolRecognitionCNN, which operates on a 128×128 grayscale image input.

Data Pipeline (dataset_flowchart_code_final.mmd)
The data is processed via a highly optimized TensorFlow pipeline:

Input: RGB images are loaded.

Conversion: A Lambda layer immediately converts the image to Grayscale (1 channel).

Normalization: Pixel values are scaled to [0,1].

Optimization: Data is shuffled, cached, and prefetched for GPU performance.

CNN Model (cnn_symbol_recognition.py)
The network uses three convolutional blocks:

Block

Layers

Key Technique

Purpose

Input

Input → rgb_to_grayscale → data_augmentation

Data Augmentation

Creates variability in training set.

Features

3x Conv2D + BatchNormalization + MaxPool

Batch Normalization

Stabilizes training and improves convergence speed.

CAM Anchor

conv3_cam_layer (128 filters)

Grad-CAM Source

Provides the final feature map for visualization.

Classifier

GlobalAveragePooling2D → Dense (15)

Softmax Activation

Produces the final probability distribution across 15 classes.

🛠️ Setup and Requirements
Prerequisites
Python 3.8+

TensorFlow / Keras 2.x

NumPy, Matplotlib

Scikit-learn (for classification report)

Dataset: Download the "Hand-drawn Electric circuit Schematic Components" dataset from Kaggle and ensure the unzipped folder structure (SolvaDataset_200_v3 containing 15 class subfolders) is in the root directory.

Running the Code
The project is structured into six cells for easy execution in a Jupyter or Google Colab notebook.

Cell 1: Imports and Configuration (Sets EPOCHS=30).

Cell 2: Data Preparation (Loads and preprocesses train_ds and val_ds).

Cell 3: Model Definition and Summary (Defines and prints the model architecture).

Cell 4: Model Training and Saving (model.fit and saves to circuit_symbol_cnn_model.h5).

Cell 5: Evaluation, History Plotting, and Grad-CAM Functions (Generates classification report).

Cell 6: Run Visualization (Executes the display_cam function to show a sample image and its corresponding heat map).

📊 Results and Visualization
After training, the notebook outputs the following:

Accuracy/Loss Plots: Visual representation of training stability and convergence.

Classification Report: Detailed per-class metrics (Precision, Recall, F1-Score) to identify the hardest-to-classify symbols.

Grad-CAM Visualization: A side-by-side display of a validation image and its heatmap, confirming that the model focuses on the core structure of the schematic symbol when making a prediction.
