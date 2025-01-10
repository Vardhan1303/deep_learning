# Fashion MNIST Image Classification with Keras 🧵👚👖

### 🏆 Part of the Udemy Course: [Python for Computer Vision with OpenCV and Deep Learning](https://www.udemy.com/certificate/UC-537bb748-f614-4175-803c-3dcc6529c1a7/)

This repository contains a deep learning model built using **Keras** and **Convolutional Neural Networks (CNNs)** to classify images from the **Fashion MNIST** dataset. The Fashion MNIST dataset consists of grayscale 28x28 images of 10 types of clothing items, and the goal of this project is to train a model that can classify these images accurately.

## 🎯 Project Overview

The model performs image classification on the **Fashion MNIST** dataset. It uses a **Convolutional Neural Network (CNN)** architecture with the following layers:

- 🧠 A **2D Convolutional Layer** with 32 filters and kernel size of (4,4)
- 🔲 A **Pooling Layer** with a pool size of (2,2)
- 🔄 A **Flatten Layer** to reshape the data before passing it to the fully connected layers
- 💡 A **Dense Hidden Layer** with 128 neurons and ReLU activation
- 🎯 A final **Dense Layer** with 10 neurons and a softmax activation function for multi-class classification

## 🗂️ Dataset

The Fashion MNIST dataset consists of **60,000 training images** and **10,000 test images**. Each image is **28x28 pixels**, and each image belongs to one of the following 10 classes:

| Label | Description   |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

## ⚙️ Requirements

To run this project, you'll need to install the following libraries:

- **TensorFlow** (includes Keras)
- **NumPy**
- **Matplotlib**
- **scikit-learn**

You can install these dependencies using `pip`:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 🛠️ Model Architecture

The model is built using Keras' Sequential API, with the following layers:

```Python code
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu'))

# Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Dense Hidden Layer
model.add(Dense(128, activation='relu'))

# Final Dense Layer for classification
model.add(Dense(10, activation='softmax'))
```

## 📝 Model Compilation
The model is compiled using the following parameters:

Loss function: categorical_crossentropy
Optimizer: rmsprop
Metrics: accuracy

## 🚀 Training
The model is trained for 10 epochs using the training data (x_train, y_train), with the following code:

```Python code
model.fit(x_train, y_cat_train, epochs=10)
```

## 📊 Evaluation
After training the model, you can evaluate it on the test set and obtain a classification report that includes precision, recall, and F1-score for each class:

```Python code
Copy code
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_classes))
```

## 🏅 Example Classification Report:
 
```yaml
precision    recall  f1-score   support

           0       0.86      0.85      0.86      1000
           1       0.99      0.98      0.98      1000
           2       0.82      0.87      0.85      1000
           3       0.93      0.91      0.92      1000
           4       0.88      0.83      0.86      1000
           5       0.98      0.98      0.98      1000
           6       0.74      0.76      0.75      1000
           7       0.96      0.97      0.97      1000
           8       0.98      0.98      0.98      1000
           9       0.97      0.96      0.97      1000

accuracy                           0.91     10000
macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000
```

The model achieves an accuracy of 91% on the test dataset.

## 🎓 Udemy Course Certificate
This project was completed as part of a **Udemy course** on **Deep Learning for Image Classification**.

#### 📜 [Click here to view my Udemy certificate](https://www.udemy.com/certificate/UC-537bb748-f614-4175-803c-3dcc6529c1a7/)

## 🤝 Credits

- Instructor: Jose Portilla
- Course: Python for Computer Vision with OpenCV and Deep Learning

## 🛠️ Skills Demonstrated

- 🧵 **Image classification** using **Convolutional Neural Networks (CNNs)** on the **Fashion MNIST** dataset
- 🖼️ **Preprocessing grayscale images** and reshaping data for CNN input
- 📊 **Training and evaluating** the model with performance metrics like accuracy, precision, recall, and F1-score
- ⚡ **Model compilation** using **categorical cross-entropy loss** and **RMSprop optimizer**
- 🔄 **Hyperparameter tuning** (e.g., number of filters, kernel size, number of neurons)
- 📉 **Visualizing model performance** over epochs with accuracy and loss curves

⭐ If you like this project, give it a star! 🌟