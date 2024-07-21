## Step 1. Import libraries


```python
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math 
import copy
import pickle
import gc

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler
```

## Step 1. Load and explore the data


```python
TARGET = 'Response'
SEED = 94
```


```python
print('Loading Data...')
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

submission_data = pd.read_csv('input/sample_submission.csv')

print('Data Load Successfully.')
```

    Loading Data...
    Data Load Successfully.
    


```python
train.shape, test.shape
```




    ((11504798, 12), (7669866, 11))




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11504798 entries, 0 to 11504797
    Data columns (total 12 columns):
     #   Column                Dtype  
    ---  ------                -----  
     0   id                    int64  
     1   Gender                object 
     2   Age                   int64  
     3   Driving_License       int64  
     4   Region_Code           float64
     5   Previously_Insured    int64  
     6   Vehicle_Age           object 
     7   Vehicle_Damage        object 
     8   Annual_Premium        float64
     9   Policy_Sales_Channel  float64
     10  Vintage               int64  
     11  Response              int64  
    dtypes: float64(3), int64(6), object(3)
    memory usage: 1.0+ GB
    


```python
train.describe()
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
      <th>id</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
      <td>1.150480e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.752398e+06</td>
      <td>3.838356e+01</td>
      <td>9.980220e-01</td>
      <td>2.641869e+01</td>
      <td>4.629966e-01</td>
      <td>3.046137e+04</td>
      <td>1.124254e+02</td>
      <td>1.638977e+02</td>
      <td>1.229973e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.321149e+06</td>
      <td>1.499346e+01</td>
      <td>4.443120e-02</td>
      <td>1.299159e+01</td>
      <td>4.986289e-01</td>
      <td>1.645475e+04</td>
      <td>5.403571e+01</td>
      <td>7.997953e+01</td>
      <td>3.284341e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>2.000000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.630000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+01</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.876199e+06</td>
      <td>2.400000e+01</td>
      <td>1.000000e+00</td>
      <td>1.500000e+01</td>
      <td>0.000000e+00</td>
      <td>2.527700e+04</td>
      <td>2.900000e+01</td>
      <td>9.900000e+01</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.752398e+06</td>
      <td>3.600000e+01</td>
      <td>1.000000e+00</td>
      <td>2.800000e+01</td>
      <td>0.000000e+00</td>
      <td>3.182400e+04</td>
      <td>1.510000e+02</td>
      <td>1.660000e+02</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.628598e+06</td>
      <td>4.900000e+01</td>
      <td>1.000000e+00</td>
      <td>3.500000e+01</td>
      <td>1.000000e+00</td>
      <td>3.945100e+04</td>
      <td>1.520000e+02</td>
      <td>2.320000e+02</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.150480e+07</td>
      <td>8.500000e+01</td>
      <td>1.000000e+00</td>
      <td>5.200000e+01</td>
      <td>1.000000e+00</td>
      <td>5.401650e+05</td>
      <td>1.630000e+02</td>
      <td>2.990000e+02</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Combine daataset for processing
train['is_train'] = 1
test['is_train'] = 0

df = pd.concat([train, test])
df.head()
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
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
      <th>is_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>65101.0</td>
      <td>124.0</td>
      <td>187</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>43</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>58911.0</td>
      <td>26.0</td>
      <td>288</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>25</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>38043.0</td>
      <td>152.0</td>
      <td>254</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>2630.0</td>
      <td>156.0</td>
      <td>76</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Female</td>
      <td>36</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>31951.0</td>
      <td>152.0</td>
      <td>294</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Step 2. Data preprocessing


```python
# Check missing values
df.isnull().sum()
```




    id                            0
    Gender                        0
    Age                           0
    Driving_License               0
    Region_Code                   0
    Previously_Insured            0
    Vehicle_Age                   0
    Vehicle_Damage                0
    Annual_Premium                0
    Policy_Sales_Channel          0
    Vintage                       0
    Response                7669866
    is_train                      0
    dtype: int64



* **Age and Vehicle_Age (0.77)**:
    Strong positive correlation. Older individuals tend to have older vehicles.
* **Previously_Insured and Vehicle_Damage (-0.84)**:
    Strong negative correlation. If someone is previously insured, their vehicle is less likely to be damaged.
* **Policy_Sales_Channel and Age (-0.60)**:
    Moderate negative correlation. Younger individuals are more likely to be reached through certain sales channels.


```python
def transform_categorical_features(df):
    print('Transforming categorical features..')

    gender_map = {'Male': 0, 'Female': 1}
    vehicle_age = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2} 
    vehicle_damage = {'No':0, 'Yes':1}

    df['Gender'] = df['Gender'].map(gender_map)
    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age)
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vehicle_damage)

    print("Transformed successfully.")
    return df
```


```python
def create_additional_features(df):
    print('Creating additional features..')
    
    df['Vehicle_Age_Policy_Sales_Channel'] = pd.factorize(df['Vehicle_Age'].astype(str) + df['Policy_Sales_Channel'].astype(str))[0]
    df['Age_Vehicle_Age'] = pd.factorize(df['Age'].astype(str) + df['Vehicle_Age'].astype(str))[0]
    df['Prev_Insured_Vehicle_Damage'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Vehicle_Damage'].astype(str))[0]
    df['Prev_Insured_Vintage'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Vintage'].astype(str))[0]
    df['Policy_Sales_Channel_Age'] = pd.factorize(df['Policy_Sales_Channel'].astype(str) + df['Age'].astype(str))[0]

    return df
```


```python
def adjust_data_types(df):
    print('Adjusting data types')
    df['Region_Code'] = df['Region_Code'].astype(int)
    df['Annual_Premium'] = df['Annual_Premium'].astype(int)
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)
    
    return df
```


```python
def optimize_memory_usage(df):
    print('Optimizing memory usage')
    start_mem_usage = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type.name in ['category', 'object']:
            raise ValueError(f"Column '{col}' is of type '{col_type.name}'")

        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem_usage = df.memory_usage().sum() / 1024**2
    print(f'------ Memory usage before: {start_mem_usage:.2f} MB')
    print(f'------ Memory usage after: {end_mem_usage:.2f} MB')
    print(f'------ Reduced memory usage by {(100 * (start_mem_usage - end_mem_usage) / start_mem_usage):.1f}%')

    return df
```


```python
df.head()
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
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
      <th>is_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>65101.0</td>
      <td>124.0</td>
      <td>187</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>43</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>58911.0</td>
      <td>26.0</td>
      <td>288</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>25</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>38043.0</td>
      <td>152.0</td>
      <td>254</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>2630.0</td>
      <td>156.0</td>
      <td>76</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Female</td>
      <td>36</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>31951.0</td>
      <td>152.0</td>
      <td>294</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = transform_categorical_features(df)
df = adjust_data_types(df)  
df = create_additional_features(df)
df = optimize_memory_usage(df)

df.head()                          
```

    Transforming categorical features..
    Transformed successfully.
    Adjusting data types
    Creating additional features..
    Optimizing memory usage
    ------ Memory usage before: 2560.09 MB
    ------ Memory usage after: 713.17 MB
    ------ Reduced memory usage by 72.1%
    




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
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
      <th>is_train</th>
      <th>Vehicle_Age_Policy_Sales_Channel</th>
      <th>Age_Vehicle_Age</th>
      <th>Prev_Insured_Vehicle_Damage</th>
      <th>Prev_Insured_Vintage</th>
      <th>Policy_Sales_Channel_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>65101</td>
      <td>124</td>
      <td>187</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>43</td>
      <td>1</td>
      <td>28</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>58911</td>
      <td>26</td>
      <td>288</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>38043</td>
      <td>152</td>
      <td>254</td>
      <td>0.0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2630</td>
      <td>156</td>
      <td>76</td>
      <td>0.0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>36</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>31951</td>
      <td>152</td>
      <td>294</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute the correlation matrix
corr = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
```


```python
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
min_max_scaler = MinMaxScaler()

# Select features to scale
features_to_scale = ['Annual_Premium', 'Vintage', 'Policy_Sales_Channel']

# Fit and transform the selected features
df[features_to_scale] = min_max_scaler.fit_transform(df[features_to_scale])
```


```python
df.head()
```


```python
df.head()
```

## Step 4. Split the data


```python
# Split the data back into train and test sets
train = df[df['is_train'] == 1].drop(columns=['is_train'])
test = df[df['is_train'] == 0].drop(columns=['is_train'])

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = submission_data
```


```python
X_train.shape
```




    (11504798, 16)




```python
y_train.shape
```




    (11504798,)




```python
X_test.shape
```




    (7669866, 16)




```python
y_test.shape
```




    (7669866, 2)



### Subsample the data to speed up training process


```python

X_train_subsample = X_train.sample(frac=0.01, random_state=42)
y_train_subsample = y_train.sample(frac=0.01, random_state=42)

```


```python
X_test_subsample = X_test.sample(frac = 0.01, random_state=42)
```


```python
X_train_subsample.shape
X_test_subsample.shape
```


```python
X_train.head()
```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Custom callback to print additional training information
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']}")
        print(f" - loss: {logs['loss']:.4f} - auc: {logs['auc']:.4f} - val_loss: {logs['val_loss']:.4f} - val_auc: {logs['val_auc']:.4f}")


# Build the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with AUC as a metric
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, auc = model.evaluate(X_test, y_test)
print(f"Test AUC: {auc:.4f}")

# Predict probabilities
y_pred_proba = model.predict(X_test).ravel()

# Calculate the AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")
```

    Epoch 1/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m301s[0m 1ms/step - auc_2: 0.5000 - loss: 132.1440 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 2/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m250s[0m 868us/step - auc_2: 0.5003 - loss: 0.3755 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 3/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m251s[0m 872us/step - auc_2: 0.4995 - loss: 0.3732 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 4/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m258s[0m 898us/step - auc_2: 0.4995 - loss: 0.3730 - val_auc_2: 0.5000 - val_loss: 0.3725
    Epoch 5/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m264s[0m 917us/step - auc_2: 0.4997 - loss: 0.3728 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 6/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m271s[0m 940us/step - auc_2: 0.4992 - loss: 0.3731 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 7/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m270s[0m 939us/step - auc_2: 0.4998 - loss: 0.3727 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 8/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m264s[0m 918us/step - auc_2: 0.4998 - loss: 0.3733 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 9/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m261s[0m 906us/step - auc_2: 0.4993 - loss: 0.3727 - val_auc_2: 0.5000 - val_loss: 0.3722
    Epoch 10/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m265s[0m 920us/step - auc_2: 0.4994 - loss: 0.3725 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 11/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m270s[0m 936us/step - auc_2: 0.4999 - loss: 0.3727 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 12/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m257s[0m 894us/step - auc_2: 0.4999 - loss: 0.3728 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 13/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m256s[0m 888us/step - auc_2: 0.4999 - loss: 0.3726 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 14/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m248s[0m 862us/step - auc_2: 0.4998 - loss: 0.3729 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 15/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m250s[0m 867us/step - auc_2: 0.5002 - loss: 0.3728 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 16/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m251s[0m 872us/step - auc_2: 0.4998 - loss: 0.3729 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 17/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m251s[0m 871us/step - auc_2: 0.4995 - loss: 0.3727 - val_auc_2: 0.5000 - val_loss: 0.3724
    Epoch 18/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m253s[0m 878us/step - auc_2: 0.4999 - loss: 0.3730 - val_auc_2: 0.5000 - val_loss: 0.3723
    Epoch 19/100
    [1m287620/287620[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m253s[0m 879us/step - auc_2: 0.5002 - loss: 0.3729 - val_auc_2: 0.5000 - val_loss: 0.3723
    


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    Cell In[22], line 32
         29 history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])
         31 # Evaluate the model
    ---> 32 loss, auc = model.evaluate(X_test, y_test)
         33 print(f"Test AUC: {auc:.4f}")
         35 # Predict probabilities
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\utils\traceback_utils.py:122, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        119     filtered_tb = _process_traceback_frames(e.__traceback__)
        120     # To get the full stack trace, call:
        121     # `keras.config.disable_traceback_filtering()`
    --> 122     raise e.with_traceback(filtered_tb) from None
        123 finally:
        124     del filtered_tb
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\tensorflow\python\eager\execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         51 try:
         52   ctx.ensure_initialized()
    ---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         54                                       inputs, attrs, num_outputs)
         55 except core._NotOkStatusException as e:
         56   if name is not None:
    

    InvalidArgumentError: Graph execution error:
    
    Detected at node UnsortedSegmentSum_1 defined at (most recent call last):
      File "<frozen runpy>", line 198, in _run_module_as_main
    
      File "<frozen runpy>", line 88, in _run_code
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel_launcher.py", line 18, in <module>
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\traitlets\config\application.py", line 1075, in launch_instance
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\kernelapp.py", line 739, in start
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\tornado\platform\asyncio.py", line 205, in start
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\asyncio\base_events.py", line 641, in run_forever
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\asyncio\base_events.py", line 1987, in _run_once
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\asyncio\events.py", line 88, in _run
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\kernelbase.py", line 545, in dispatch_queue
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\kernelbase.py", line 534, in process_one
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\kernelbase.py", line 437, in dispatch_shell
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\ipkernel.py", line 362, in execute_request
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\kernelbase.py", line 778, in execute_request
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\ipkernel.py", line 449, in do_execute
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\ipykernel\zmqshell.py", line 549, in run_cell
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\interactiveshell.py", line 3075, in run_cell
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\interactiveshell.py", line 3130, in _run_cell
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\async_helpers.py", line 129, in _pseudo_sync_runner
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\interactiveshell.py", line 3334, in run_cell_async
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\interactiveshell.py", line 3517, in run_ast_nodes
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\IPython\core\interactiveshell.py", line 3577, in run_code
    
      File "C:\Users\user\AppData\Local\Temp\ipykernel_14524\278436547.py", line 32, in <module>
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\utils\traceback_utils.py", line 117, in error_handler
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\backend\tensorflow\trainer.py", line 425, in evaluate
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\backend\tensorflow\trainer.py", line 161, in one_step_on_iterator
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\backend\tensorflow\trainer.py", line 150, in one_step_on_data
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\backend\tensorflow\trainer.py", line 87, in test_step
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\trainers\trainer.py", line 412, in compute_metrics
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\trainers\compile_utils.py", line 330, in update_state
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\trainers\compile_utils.py", line 17, in update_state
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\metrics\confusion_metrics.py", line 1379, in update_state
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\metrics\metrics_utils.py", line 481, in update_confusion_matrix_variables
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\metrics\metrics_utils.py", line 277, in _update_confusion_matrix_variables_optimized
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\ops\math.py", line 59, in segment_sum
    
      File "C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\backend\tensorflow\math.py", line 17, in segment_sum
    
    data.shape = [64] does not start with segment_ids.shape = [32]
    	 [[{{node UnsortedSegmentSum_1}}]] [Op:__inference_one_step_on_iterator_26047228]


## Step 5. Train and evaluate the model


```python
def train_and_evaluate(model, X, y, X_test, folds=10, random_state=None):
    print(f'Training {model.__class__.__name__}\n')
    
    scores = []
    feature_importances = np.zeros(X.shape[1])
    evaluation_history = []
    
    oof_pred_probs = np.zeros(X.shape[0])
    test_pred_probs = np.zeros(X_test.shape[0])
    
    skf = StratifiedKFold(n_splits=10, random_state=94, shuffle=True)
    
    for fold_index, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model_clone = copy.deepcopy(model)
        model_clone.fit(
                X_train, 
                y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=500)
        
        feature_importances += model_clone.feature_importances_ / folds
        evaluation_history.append(model_clone.evals_result())
        
        y_pred_probs = model_clone.predict_proba(X_val)[:, 1]
        oof_pred_probs[val_index] = y_pred_probs
        
        temp_test_pred_probs = model_clone.predict_proba(X_test)[:, 1]
        test_pred_probs += temp_test_pred_probs / folds
        
        auc_score = roc_auc_score(y_val, y_pred_probs)
        scores.append(auc_score)
        
        print(f'\n--- Fold {fold_index + 1} - AUC: {auc_score:.5f}\n\n')
        
        del model_clone
        gc.collect()
    
    print(f'------ Average AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}\n\n')

    return oof_pred_probs, test_pred_probs
```


```python
best_params = {
    'alpha': 1.302348865795227e-06, 
    'max_depth': 15, 
    'learning_rate': 0.061800451723613786, 
    'subsample': 0.7098803046786328, 
    'colsample_bytree': 0.2590672912533101, 
    'min_child_weight': 10, 
    'gamma': 0.8399887056014855, 
    'reg_alpha': 0.0016943548302122801, 
    'max_bin': 71284,
    'early_stopping_rounds': 50
}
best_xgb_model = XGBClassifier(**best_params, n_estimators=12000, random_state=94, eval_metric="auc")

# Call train_and_evaluate function with XGBClassifier model
oof_pred_probs, predictions = train_and_evaluate(best_xgb_model, X_train, y_train, X_test, folds=10, random_state=SEED)
```


```python
submission = pd.DataFrame({
    'id': X_test['id'],
    'Response': predictions
})
submission.to_csv('submission.csv', index=False)
submission.head()
```

## Possible improvements


```python
##Binning some features

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Binning Vintage
bins_vintage = [0, 200, 400, 600, 800, float('inf')]
labels_vintage = ['Very New', 'New', 'Moderately New', 'Experienced', 'Very Experienced']
df['Vintage_Binned'] = pd.cut(df['Vintage'], bins=bins_vintage, labels=labels_vintage)
# Binning Annual_Premium
bins_premium = [0, 10000, 30000, 50000, 100000, float('inf')]
labels_premium = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
df['Annual_Premium_Binned'] = pd.cut(df['Annual_Premium'], bins=bins_premium, labels=labels_premium)

# Encoding Policy_Sales_Channel
le = LabelEncoder()
df['Policy_Sales_Channel_Encoded'] = le.fit_transform(df['Policy_Sales_Channel'])

# Dropping original columns
df = df.drop(['Vintage', 'Annual_Premium', 'Policy_Sales_Channel'], axis=1)

df['Annual_Premium_Binned_Numeric'], _ = pd.factorize(df['Annual_Premium_Binned'])
df['Vintage_Binned_Numeric'],_ = pd.factorize(df['Vintage_Binned'])
```


```python
##Using RandomizedSearch - hyperparamiters tunning

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


xgb_params = {
    'colsample_bylevel': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'colsample_bynode': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.5, 0.6051, 1],
    'max_bin': [256, 512, 682, 1024],
    'max_delta_step': [0, 1, 5, 7, 10],
    'max_depth': [3, 5, 10, 20, 50, 68, 100],
    'min_child_weight': [1, 3, 5, 7, 10],
    'n_estimators': [100, 500, 1000, 5000, 10000],
    'reg_alpha': [0, 0.1, 0.4651, 0.5],
    'reg_lambda': [0, 0.1, 0.5, 1],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# Set up cross-validation strategy
cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

xgb_model = XGBClassifier(objective="binary:logistic", n_jobs=-1, random_state=SEED, eval_metric="auc", verbosity=0, tree_method='hist')

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_params, n_iter=5, scoring='roc_auc', cv=cv, verbose=1, random_state=SEED)
print(random_search)
random_search.fit(X_train_subsample, y_train_subsample)

print("Best parameters found: ", random_search.best_params_)
print("Best AUC score: ", random_search.best_score_)
```
