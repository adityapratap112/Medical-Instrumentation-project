# Lung Cancer Classification using Deep Learning

![Lung Cancer Classification](rnn2/cnn_visualization.png)

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying lung cancer CT scan images into three categories: Benign cases, Malignant cases, and Normal cases. The model achieves high accuracy in distinguishing between these categories, providing a potential tool for assisting medical professionals in lung cancer diagnosis.

## Dataset
The project uses the IQ-OTHNCCD lung cancer dataset, which contains CT scan images of lungs categorized into three classes:
- Benign cases
- Malignant cases
- Normal cases

The dataset is accessed via Kaggle using the kagglehub API.

## Model Architecture
The model is a custom CNN with the following architecture:

![CNN Architecture](rnn2/cnn_architecture_diagram.png)

The architecture includes:
- 4 convolutional layers with increasing filter sizes (32, 64, 128, 256)
- Batch normalization after each convolutional layer
- Max pooling layers to reduce spatial dimensions
- Dropout layers with increasing rates (0.2, 0.3, 0.4, 0.5) for regularization
- 2 fully connected layers (512 units and 3 output units)

## Results
The model achieves high accuracy on the test set, with detailed performance metrics visualized below:

### Training Metrics
![Training Metrics](rnn2/training_metrics.png)

### Confusion Matrix
![Confusion Matrix](rnn2/confusion_matrix.png)

### ROC Curves
![ROC Curves](rnn2/roc_curves.png)

### Prediction Samples
![Prediction Samples](rnn2/prediction_samples.png)

## Feature Maps Visualization
The project includes visualization of feature maps from the convolutional layers:

![Feature Maps](rnn2/cnn_visualization.png)

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn
- Kagglehub

### Installation
1. Clone the repository:
```bash
git clone https://github.com/adityapratap112/Medical-Instrumentation-project.git
cd Medical-Instrumentation-project
```

2. Install the required packages:
```bash
pip install torch torchvision opencv-python matplotlib pandas scikit-learn kagglehub
```

3. (Optional) For model visualization:
```bash
pip install torchviz
```

## Usage

### Training the Model
To train the model from scratch:
1. Open the `rnn2/train.ipynb` notebook in Jupyter or Google Colab
2. Run all cells to:
   - Download the dataset
   - Preprocess the images
   - Train the CNN model
   - Evaluate performance
   - Generate visualizations

### Using the Demo
To use the pre-trained model for predictions:
1. Open the `rnn2/demo.ipynb` notebook
2. Run all cells to:
   - Load the pre-trained model
   - Make predictions on sample images
   - Visualize the results
   - Explore the model architecture

### Making Predictions on New Images
The demo notebook includes a user-friendly function for making predictions:
```python
result = predict_lung_cancer(image_path)
```

## Project Structure
```
Medical-Instrumentation-project/
├── rnn2/
│   ├── demo.ipynb                    # Demo notebook for using the trained model
│   ├── train.ipynb                   # Training notebook
│   ├── lung_cnn_full_model.pth       # Full saved model
│   ├── lung_cnn_state_dict.pth       # Model state dictionary
│   ├── lung_rnn_best_model.pth       # Best RNN model
│   ├── lung_rnn_full_model.pth       # Full RNN model
│   ├── lung_rnn_state_dict.pth       # RNN model state dictionary
│   ├── cnn_architecture_diagram.png  # CNN architecture visualization
│   ├── cnn_visualization.png         # Feature maps visualization
│   ├── confusion_matrix.png          # Confusion matrix visualization
│   ├── model_architecture.png        # Model architecture graph
│   ├── prediction_samples.png        # Sample predictions
│   ├── roc_curves.png                # ROC curves
│   └── training_metrics.png          # Training metrics visualization
└── README.md                         # Project documentation
```

## Acknowledgments
- The IQ-OTHNCCD lung cancer dataset from Kaggle
- PyTorch and torchvision libraries
- Kagglehub for dataset access

## License
This project is licensed under the MIT License - see the LICENSE file for details.