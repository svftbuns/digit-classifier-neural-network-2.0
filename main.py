import pandas as pd
import numpy as np

digit=pd.read_csv('train.csv')
digit=np.array(digit) # Change to numpy array
m,n = digit.shape # m is the number of rows, n is the number of columns
np.random.shuffle(digit)

# Split up labels and pixels
labels=digit[:,0]
pixels=digit[:,1:]

X=pixels.reshape(len(pixels),1,28,28)
X=X/255.0 # Normalization

x_train=X[1000:]
y_train=labels[1000:]
x_test=X[:1000]
y_test=labels[:1000]

# Creating the ConvolutionalNN instance
CNN = ConvolutionalNN(layers=[
    Convolutional((1, 28, 28), kernel_size=3, depth=3, activation='Sigmoid'),  # Output shape: (3, 26, 26)
    MaxPooling(pool_size=2, stride=2),  # Output shape: (3, 13, 13)
    Convolutional((3, 13, 13), kernel_size=3, depth=5, activation='Sigmoid'),  # Output shape: (5, 11, 11)
    MaxPooling(pool_size=2, stride=2),  # Output shape: (5, 5, 5)
    Dropout(0.2),
    Dense(125, 10, activation='Softmax')  # Flatten shape: (5*5*5, 1) = (125, 1)
])


train_loss, val_accuracy=CNN.train(x_train,y_train,validation=[x_test,y_test],epochs=10,learning_rate=0.05)

""" Output: 
Epoch 1/10, Train Loss=0.19636010399636816, Val Accuracy=0.51, Time Elapsed=6.414 mins
Epoch 2/10, Train Loss=0.15635710399771735, Val Accuracy=0.64, Time Elapsed=6.285 mins
Epoch 3/10, Train Loss=0.12900347839776877, Val Accuracy=0.643, Time Elapsed=6.3632 mins
Epoch 4/10, Train Loss=0.1108079826528142, Val Accuracy=0.713, Time Elapsed=6.0152 mins
Epoch 5/10, Train Loss=0.09783930575796077, Val Accuracy=0.76, Time Elapsed=5.9867 mins
Epoch 6/10, Train Loss=0.08891628745103491, Val Accuracy=0.763, Time Elapsed=5.9651 mins
Epoch 7/10, Train Loss=0.08387877917075043, Val Accuracy=0.756, Time Elapsed=5.9915 mins
Epoch 8/10, Train Loss=0.07979329207067799, Val Accuracy=0.779, Time Elapsed=5.9574 mins
Epoch 9/10, Train Loss=0.07593468929775744, Val Accuracy=0.768, Time Elapsed=5.9968 mins
Epoch 10/10, Train Loss=0.07345111020646794, Val Accuracy=0.807, Time Elapsed=9.0547 mins 

Kaggle Test Score: 0.80657
"""
