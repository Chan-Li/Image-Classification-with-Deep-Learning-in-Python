# From-Scratch Deep Learning Models for Image Classification

This repository contains a collection of from-scratch implementations of common deep learning architectures — including MLPs, CNNs, RNNs, and Hopfield-type networks — written in pure Python.  
The goal is to build a transparent, minimal framework that helps illustrate how learning, optimization, and feature extraction occur inside neural networks without relying on external libraries such as PyTorch or TensorFlow.

Although simple by design, these implementations demonstrate the core components needed in real computer vision workflows, such as dataset loading, preprocessing, training loops, gradient-based optimization, and evaluation.

---

## ✨ What This Repository Demonstrates
- End-to-end **image classification pipelines** implemented from scratch  
- Manual construction of:
  - forward and backward passes  
  - activation functions  
  - loss computation  
  - parameter updates (SGD/variants)  
- Implementation of classic CNN architectures (e.g., LeNet) for vision tasks  
- Training on widely used benchmark datasets (MNIST, CIFAR-10)  
- Clear exposure of underlying learning dynamics, model behavior, and limitations  

These components form the foundation of more advanced deep learning models used in industrial inspection, anomaly detection, and scientific imaging.

---

## 📦 Datasets

This repository includes loaders for several standard computer vision datasets used in classification tasks:

### **MNIST**
Handwritten-digit dataset (60k train / 10k test).  
Used to test MLP and small CNN architectures.

Loading example:
```python
from Datasets import load_MNIST as load2
mnist = load2.load_mnist(one_hot=True)
train_data = mnist[0][0][0:60000].T
train_label = mnist[0][1][0:60000].T
test_data  = mnist[1][0][0:10000].T
test_label = mnist[1][1][0:10000].T
