
# Module 13: Risk Management Associate Using Neural Networks with Tensorflow and Keras

*"A FinTech project as a 'Risk Management Associate' to Predict Successful Startups at a Venture Capital Company."*

---

## Background

This project constructs a Neural Network(NN) program to predict which startup investments will be successful for a venture capital firm. Using historical data of 34,000 funding deals certain features are used to construct a binary classifier model to make predictions for funding successful business ventures. In order to construct a NN model, Tensorflow and Keras are used from the python library to build a NN deep learning program employing algorithms. Using an algorithm optimizer function to shape the NN on trained data, it molds a model that reduces losses in order to make more accurate predictions. This NN is compiled to test different parameters and then evaluate its binary classification model accuracy that predicts if funding for startups will be successful.   

With the exponential growth of novel technology discoveries in several industries to make a better world, many new innovations are being forged into implementation by startup companies to apply their discoveries. Applying these innovations requires capital funding to execute production for public utility. Most funding comes from private investors and venture capital(VC) firms willing to risk capital on novel unproven ideas for respective industry markets. To mitigate investment losses a financial application model is utilized to preprocess data for a NN, then compile it, train it, evaluate it and then optimize a NN model. The optimization process seeks to improve the NN model's accuracy score and lower the loss calculation. Using Keras, the NN models parameters, layer configuration, input weights, activation functions and evaluation metrics are saved in a Hierarchical DataFormat HDF5Links to an external site file for future analysis and reference.   

 The FinTech app technology in this program utilizes TensorFlow's open source platform for machine learning that allows code to run across platforms more efficiently. And Keras (the deep learning framework) from the Python library serves as an abstract of TensorFlow to simplify the coding process. This allows more time to focus on tuning the optimization of NN models and evaluating them to solve business problems. The purpose of using deep learning NN models in this case is to experiment with several algorithms models and test the performance of binary classification outcomes. In this case, we are looking for increasing probability of investing in successful start-ups, while minimizing capital risk in the predictions.  
 
 A successful VC firm that mitigates capital risk is more aptly able to raise greater investor interest and increase amount of funding to ensure more successful startup chances. 

---
## Evaluation Results

The following evaluation describes the performance test of the imported models, with the loss and accuracy metric scores of all four neural network machine learning models.

* NN Original Model: 
  * Original nn features: 2 hiddenlayers 'relu', with 50epochs
  * Original nn Model Accuracy: 0.7289
  * Original nn Model Loss: 0.5561     Where '0' represents a 'healthy loan' and, '1' represents a high-risk-loan for the scores. 
  
* NN Alternative Model 1 Results:
  * nn_Al features: 1 hiddenlayer 'relu', with 50epochs       
  * nn_Al Accuracy: 0.7310
  * nn_Al Loss: 0.5590

* NN Alternative Model 2 Results:
  * nn_A2 features: 1 hiddenlayer 'relu', with 100epochs      
  * nn_A2 Accuracy: 0.7287
  * nn_A2 Loss: 0.5660 

* NN Alternative Model 3 Results:
  * nn_A3 features- 2 hiddenlayer 'relu' + 1 hiddenlayer 'tanh' , with 50epochs       
  * nn_A3 Accuracy: 0.7296
  * nn_A3 Loss: 0.5508 

---

## Technologies

The software operates on python 3.9 with the installation package imports embedded with Anaconda3 installation. Pandas and scikitlearn are imported libraries for this program. Additional application tools that you need for this module include TensorFlow 2.0 library that should already be installed in the default Conda environmentand. Keras is the popular deep learning framework that serves as a high-level API for TensorFlow 2.0. Pleaase reference the latest official [TensorFlow Install Guide](https://www.tensorflow.org/install/pip) to troubleshoot issues.   

* [anaconda3](https://docs.anaconda.com/anaconda/install/windows/e) 

* [sklearn](https://scikit-learn.org/stable/install.html) 

* [TensorFlow](https://www.tensorflow.org/) 

* [Keras](https://keras.io/getting_started/) . 
---

## Installation Guide

Before running the applications first navigate to [TensorFlow](https://www.tensorflow.org/install/pip#windows) for installation instructions. Then verify if the installation as been completed. Using `python -c "import tensorflow as tf;print(tf.__version__)"`. Keras is included with TensorFlow 2.0 but still needs verification prior to use `python -c "import tensorflow as tf;print(tf.keras.__version__)"`The output should show version 2.2.4-tf or later. 


```python libraries
pip install --upgrade tensorflow
python3 -m pip install tensorflow
from tensorflow import keras
```
```from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder 
```

---
# Usage

This application is launched from web-based Jupyter notebook utilizing Pandas and scikitlearn `StandardScaler` to preprocess the data, along with `OneHotEncoder` to encode categorical variables for the NN model computations. TensorFlow's Keras `Sequential` framework sets the design of the NN layer structure so data can flow sequentially between each layer. It's `Dense`module allows additional NN layers to be added to the model framework. TensorFlow is used to compile a NN model using `binary_crossentropy` function,`adam` optimizer and `accuracy` evaluation metrics.    

The program is developed in Jupyter notebook on a jupyter **.ipny** file. The Python library makes it possible to utilize TensorFlow and Keras build this NN deep learning machine algorithm. The design applies the model-fit-predict process to make a binary classification of whether a startup is successful or not.
 

![NN Model Evals: Origin & A1](Images\Screenshot2022-06-15032835.png) 

![NN Model Evals: A2 & A3](Images\Screenshot2022-06-15033615.png) 



```python
venture_funding_with_deep_learning.ipynb
```
 

---

## Contributors

*Provided to you by digi-Borg FinTek*, 
Dana Hayes: nydane1@gmail.com

---

## License

Columbia U. Engineering


>>>>>>> b9d57d6932c7a2955ea36e7cb11f70e9be46f990
