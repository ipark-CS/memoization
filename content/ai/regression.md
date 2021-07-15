+++
author = "ipark"
title = "Regression using Neural Network in Keras (Boston, Hyderabad dataset)"
date =  2021-06-26
draft =  false
type = "ai"
layout = "ai"
description = ""
tags = ["DL", "Deep Learning", "regression", "keras"
]
+++
## [Case1] Boston Housing Price Dataset

### 0)  Set Random Seed for Later Reproducibility


```python
#https://keras.io/ko/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(42) # For starting core Python generated random numbers 
rn.seed(12345) # Random number generation in the TensorFlow backend 
tf.random.set_seed(1234) # Random number generation in the TensorFlow backend 
```

### 1)  Loading the Boston housing dataset


```python
#https://keras.io/api/datasets/boston_housing/#load_data-function
#tf.keras.datasets.boston_housing.load_data(
#    path="boston_housing.npz", test_split=0.2, seed=113)
# feature, target: median values of the houses in $1,000.
from tensorflow.keras.datasets import boston_housing
(train_ft, train_tg), (test_ft, test_tg) = boston_housing.load_data()
```


```python
print(f'train:test={len(train_ft)}:{len(test_ft)}')
```

    train:test=404:102


### 2)  Normalizing the data


```python
# m=[[1,2],     +----> axis=1
#    [3,4]]    |
#             axis=0
ft_wise_mean = train_ft.mean(axis=0) # feature-wise mean
train_ft -= ft_wise_mean
ft_wise_std = train_ft.std(axis=0) # feature-wise std 
train_ft /= ft_wise_std
#
test_ft -= ft_wise_mean
test_ft /= ft_wise_std
```

### 3)  Model definition


```python
num_features = len(train_ft[1])
num_features # 13
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
def build_regression_model():
    model = Sequential()
    model.add( Dense(64, activation='relu') )
    model.add( Dense(64, activation='relu') )
    model.add( Dense(1) )
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
```

### 4)  K-fold validation + Saving the validation logs at each fold


```python
all_mae_trace = []
def k_fold_validation(k, num_epochs, num_batch):
    all_val_scores = []
    val_samples = len(train_ft)//k
    for i in range(k):
# 1) prepare validation data from partition k
        val_ft_k = train_ft[i*val_samples : (i+1)*val_samples]
        val_tg_k = train_lb[i*val_samples : (i+1)*val_samples]
# 2) prepare training data
        train_ft_k = np.concatenate(
        [ train_ft[:i*val_samples], train_ft[(i+1)*val_samples:] ],
        axis=0) # feature-wise, i.e. column-wise
        train_tg_k = np.concatenate(
        [ train_tg[:i*val_samples], train_tg[(i+1)*val_samples:] ],
        axis=0) # feature-wise, i.e. column-wise
# 3) build keras model
        model = build_regression_model()
        ################################
# 4) train model    
        trace = model.fit(train_ft_k, train_tg_k, 
            validation_data=(val_ft_k, val_tg_k), ### k-fold val
            epochs=num_epochs,
            batch_size=num_batch, verbose=0)
# 5) evaluate model on the validation set    
        val_mse, val_mae = model.evaluate(
            val_ft_k, val_tg_k, verbose=0)
        all_val_scores.append(val_mae)
    all_mae_trace.append(trace.history['val_mae'])    
    return all_val_scores
```

### 5)  Building the history of successive mean K-fold validation scores


```python
k, num_epochs, num_batch = 4, 100, 20
k_fold_validation(k, num_epochs, num_batch)
avg_mae_trace = [np.mean([x[i] for x in all_mae_trace]) \
                 for i in range(num_epochs)]
```

### 6) Plotting validation scores


```python
import matplotlib.pyplot as plt
avg_mae_trace = avg_mae_trace[10:]
plt.plot(range(1, len(avg_mae_trace)+1), avg_mae_trace)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
```




    Text(0, 0.5, 'Validation MAE')




    
<img src="/img/Boston/output_16_1.png" width="500">
    


### 8)  Fianl Training, Evaluation and Prediction


```python
# final training 
model = build_regression_model()
model.fit(train_ft, train_tg,
          epochs=130, batch_size=16, verbose=0)

# evaluate
test_mse_loss, test_mae_score = model.evaluate(test_ft, test_tg, 
                                              verbose=0)
print(f'Mean Abs Error = ${test_mae_score*1000:.2f}')

# predict
predict_vec = model.predict(test_ft)
predict_diff = np.array([abs(p-test_tg[i]) \
                         for i, p in enumerate(predict_vec)])
print(f'Mean Abs Error = ${predict_diff.mean()*1000:.2f}')
```

    Mean Abs Error = $2442.92
    Mean Abs Error = $2442.92


---
## [Case2] Hyderabad Housing Price Dataset

### 0) For reproducibility, set random seed as a practice


```python
import numpy as np
import tensorflow as tf
SEED=1
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### 1) Loading, preprocessing regression dataset


```python
import pandas as pd
data = pd.read_csv('Hyderabad.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Area</th>
      <th>Location</th>
      <th>No. of Bedrooms</th>
      <th>Resale</th>
      <th>MaintenanceStaff</th>
      <th>Gymnasium</th>
      <th>SwimmingPool</th>
      <th>LandscapedGardens</th>
      <th>JoggingTrack</th>
      <th>...</th>
      <th>LiftAvailable</th>
      <th>BED</th>
      <th>VaastuCompliant</th>
      <th>Microwave</th>
      <th>GolfCourse</th>
      <th>TV</th>
      <th>DiningTable</th>
      <th>Sofa</th>
      <th>Wardrobe</th>
      <th>Refrigerator</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6968000</td>
      <td>1340</td>
      <td>Nizampet</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29000000</td>
      <td>3498</td>
      <td>Hitech City</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6590000</td>
      <td>1318</td>
      <td>Manikonda</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5739000</td>
      <td>1295</td>
      <td>Alwal</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5679000</td>
      <td>1145</td>
      <td>Kukatpally</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in /opt/anaconda3/envs/KerasPy3.6/lib/python3.6/site-packages (0.24.2)
    Requirement already satisfied: numpy>=1.13.3 in /opt/anaconda3/envs/KerasPy3.6/lib/python3.6/site-packages (from scikit-learn) (1.19.5)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/anaconda3/envs/KerasPy3.6/lib/python3.6/site-packages (from scikit-learn) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/anaconda3/envs/KerasPy3.6/lib/python3.6/site-packages (from scikit-learn) (1.5.4)
    Requirement already satisfied: joblib>=0.11 in /opt/anaconda3/envs/KerasPy3.6/lib/python3.6/site-packages (from scikit-learn) (1.0.1)



```python
targets = data['Price'] # ground-truth to be predicted
features = data.drop(['Price', 'Location'], axis=1) # column

# normalize data
features_mean = features.mean(axis=0) # feature-wise mean
features -= features_mean
features_std = features.std(axis=0) # feature-wise std 
features /= features_std

from sklearn.model_selection import train_test_split
# spilt data into train:test=8:2
train_ft, test_ft, train_tg, test_tg =\
train_test_split(features, targets, test_size=0.2)
train_ft = train_ft.to_numpy()
train_tg = train_tg.to_numpy()
test_ft = test_ft.to_numpy()
test_tg = test_tg.to_numpy()
```


```python
print(f'train:test={len(train_ft)/len(test_ft):.2f}')
```

    train:test=4.00


### 2) Building a regression model


```python
# build a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
ncols = features.shape[1]
model =  Sequential()
model.add( Dense(ncols, activation='relu', input_shape=(ncols,)) )
model.add( Dropout(0.2) )
model.add( Dense(128, activation='relu') )
model.add( Dropout(0.2) )
model.add( Dense(64, activation='relu') )
model.add( Dropout(0.2) )
model.add( Dense(1) ) # output = no activation
```

### 3) Training


```python
# compile a model with specific optimizer, loss, and monitoring metric
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```


```python
# training with dataset, epochs, and batch_size
model.fit(train_ft, train_tg, epochs=50, batch_size=10, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x153b64fd0>



### 4) Evaluation (loss), Prediction (final answer)


```python
# evaluate
test_mse_loss, test_mae_score = model.evaluate(test_ft, test_tg, 
                                              verbose=0)
print(f'Mean Abs Error = ${test_mae_score:.2f}')

# predict
predictions = model.predict(test_ft)
predict_diff = np.array([abs(p-test_tg[i]) \
                         for i, p in enumerate(predictions)])
print(f'Mean Abs Error = ${predict_diff.mean():.2f}')
```

    Mean Abs Error = $2294626.25
    Mean Abs Error = $2294626.05

