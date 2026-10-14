# Handwritten-MNIST
# Handwritten Digit Classification using PyTorch

## 📌 Project Overview

This project implements a **fully connected neural network using PyTorch** to classify handwritten digits from 0 to 9 using Scikit-learn's built-in Digits dataset.

The dataset contains 8×8 grayscale images of handwritten digits. Each image is flattened into a 64-dimensional feature vector and standardized before being passed through the neural network for classification.

The goal is to understand the fundamentals of neural networks, including forward propagation, loss calculation, backpropagation, and optimization using PyTorch.

## 📊 Dataset

- **Dataset:** Scikit-learn Digits Dataset
- **Source:** `sklearn.datasets.load_digits()`
- **Number of samples:** 1,797
- **Image dimensions:** 8 × 8 pixels
- **Number of features:** 64
- **Number of classes:** 10 (digits 0–9)
- **Pixel intensity range:** 0–16

The dataset is split into training and testing sets using an 80:20 ratio.

## 🧠 Model Architecture

The neural network is implemented using PyTorch's `nn.Sequential`.

| Layer | Configuration |
|---|---|
| Input layer | 64 features |
| Hidden layer 1 | 32 neurons + ReLU |
| Hidden layer 2 | 16 neurons + ReLU |
| Output layer | 10 neurons |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Training epochs | 10,000 |

The output layer produces 10 logits, one for each digit class. The predicted class is obtained using `torch.argmax()`.

## ⚙️ Project Workflow

1. **Load the dataset:** Load handwritten digit images and their corresponding labels.
2. **Preprocess the data:** Scale pixel values by dividing by 16.
3. **Split the dataset:** Create training and testing sets using an 80:20 split.
4. **Standardize features:** Fit `StandardScaler` on the training data and apply it to both sets.
5. **Build the neural network:** Create a fully connected network with two hidden layers.
6. **Train the model:** Use Cross-Entropy Loss and the Adam optimizer.
7. **Evaluate performance:** Predict the test labels and calculate classification accuracy.

## 🛠️ Tech Stack

- Python
- PyTorch
- Scikit-learn
- NumPy
- Matplotlib

## 📈 Results

The model is evaluated on the held-out test dataset using classification accuracy.

**Test Accuracy:** Run the script to obtain the actual test accuracy.

The model's performance can be improved further through hyperparameter tuning, mini-batch training, and regularization techniques.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install torch scikit-learn matplotlib numpy
```

### 3. Run the Python script

Save the implementation as `main.py` and execute:

```bash
python main.py
```

The script displays a sample handwritten digit and its label, prints training loss periodically, and reports the final test accuracy.

## 📚 Concepts Learned

- Neural network architecture using PyTorch
- Data preprocessing and feature standardization
- Training and testing data separation
- Forward propagation
- Cross-Entropy Loss
- Backpropagation and gradient computation
- Adam optimization
- Multiclass classification
- Model evaluation using accuracy

## 🔮 Future Improvements

- Implement mini-batch training using `DataLoader`.
- Track training and validation loss.
- Experiment with different network architectures and learning rates.
- Evaluate precision, recall, and F1-score.
- Build a CNN for handwritten digit classification.
- Create a simple interface to predict digits from user-drawn images.

## 👨‍💻 Author

Developed as part of my journey toward understanding machine learning and deep learning using PyTorch.
