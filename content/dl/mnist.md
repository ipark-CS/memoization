+++
author = "ipark"
title = "Image Recognition (MNIST dataset) in Keras"
date =  2021-06-24T13:58:29-07:00
draft =  false
type = "dl"
layout = "dl"
description = ""
tags = ["DL", "Deep Learning", "mnist", "image recognition", "keras"
]
+++
## Image Recognition (MNIST dataset) in Keras

```python
# tensorflow random seed set to reproduce later
import tensorflow as tf
#tf.random.set_seed(1234) # poor
tf.random.set_seed(1) # poor
```

### 1) importing and reading dataset

```python
(train_ft, train_lb), (test_ft, test_lb) = \
                                        tf.keras.datasets.mnist.load_data()
                                        #################
```

```python
print(f'train_set:test_set={len(train_ft)}:{len(test_ft)}')
```
    train_set:test_set=60000:10000

```python
# plot input data
import matplotlib.pyplot as plt
fig_container = plt.figure(figsize=(20,20)) # .figure is top container
nrows = 1
ncols = 5
for i in range(ncols):
    # subplot(nrows, ncols, index, **kwargs)
    ax = fig_container.add_subplot(nrows, ncols, i+1) 
    ax.imshow(train_ft[i], cmap='Greys')
    ax.set_title('label=' + str(train_lb[i]))
```
<img src="/img/MNIST/output_5_0.png" width="500">
    
### 2) preprocessing data

```python
# input flatten
train_ft[0].shape # (28, 28)
train_ft_flat = train_ft.reshape(-1, 28*28)
test_ft_flat = test_ft.reshape(-1, 28*28)
```

```python
# categorize
num_classes = len(set(train_lb))
print(num_classes)
# one-hot-encode
train_lb_cat = tf.keras.utils.to_categorical(train_lb, num_classes) 
test_lb_cat = tf.keras.utils.to_categorical(test_lb, num_classes) 
               ##############
```
    10

### 3) building neural network model

```python
# build a model
model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Dense(128, activation='relu', \
                                      input_shape=(28*28,)) ) # input
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(64, activation='relu') )
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(num_classes, \
                                    activation='softmax') ) # output
```

```python
# compile a model with specific optimizer, loss, and monitoring metric
model.compile(optimizer=tf.keras.optimizers.Adam(), \
                       #tf.keras.optimizers.RMSprop(0.01), 
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
#model.compile( optimizer='adam', \
                loss='categorical_crossentropy', \
                metrics=['accuracy'])
model.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_20 (Dense)             (None, 128)               100480    
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_21 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 109,386
    Trainable params: 109,386
    Non-trainable params: 0
    _________________________________________________________________

### 4) training neural network

```python
# training fit to **train_set** using specific epochs & batch_size
model.fit(train_ft_flat, train_lb_cat, epochs=3, batch_size=20)
```
    Epoch 1/3
    3000/3000 [==============================] - 6s 2ms/step - loss: 2.3589 - categorical_accuracy: 0.6836
    Epoch 2/3
    3000/3000 [==============================] - 6s 2ms/step - loss: 0.6324 - categorical_accuracy: 0.8352
    Epoch 3/3
    3000/3000 [==============================] - 6s 2ms/step - loss: 0.4446 - categorical_accuracy: 0.8817

    <tensorflow.python.keras.callbacks.History at 0x15bfd2550>

### 5) evaluation model

```python
# prediction using test set
predict_vec = model.predict(test_ft_flat)
```

```python
import numpy as np
predictions = [np.argmax(p) for p in predict_vec]
```

```python
# plot prediction vs. true label
print(f'true label={test_lb[4]}')
print(f'predict label={predictions[4]}')
plt.imshow(test_ft[4], cmap='Greys')
```
    true label=4
    predict label=4

    <matplotlib.image.AxesImage at 0x15532ee80>
    
<img src="/img/MNIST/output_17_2.png" width="100">
    
```python
# accuracy
correct_num = 0
for i, p in enumerate(predictions):
    if test_lb[i] == p:
        correct_num += 1
print(f'model accuracy = {correct_num/len(predictions)}')
    
```
    model accuracy = 0.9067

