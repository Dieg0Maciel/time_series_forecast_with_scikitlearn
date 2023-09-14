# TIME SERIES FORECAST WITH SCIKIT-LEARN

## TABLE OF CONTENTS
1. OVERVIEW
2. DATA EXPLORATION AND FEATURE ENGINEERING
    * 2-1. Load data
    * 2-2. Missing data
    * 2-3. Casting to numerical
    * 2-4. Adding features
    * 2-5. Time series components
    * 2-6. Linear correlations: Heatmap and Scatterplots
3. MODEL SELECTION
4. TRAINING, EVALUATION AND PREDICTIONS 
5. CONCLUSION
6. FUTURE WORK
7. REFERENCES

## 1. OVERVIEW

In this project we will use the [Austin Weather](https://www.kaggle.com/datasets/grubenm/austin-weather) dataset provided by [Kaggle](https://www.kaggle.com/) to forecast the average temperature in Austin Texas using SciKit-Learn.

We can extend this dataset or create a dataset for another city using a weather API like [Weather API](https://www.weatherapi.com/) following the steps from the previous project [ETL PIPELINE WITH PYTHON AND AIRFLOW](https://github.com/Dieg0Maciel/etl_pipeline_with_python_and_airflow) where we studied how to build a pipeline in order to manipulate weather forecast data provided by the [Open Weather](https://openweathermap.org/) API. 


## 2. DATA EXPLORATION AND FEATURE ENGINEERIG

[Austin Weather Dataset](https://www.kaggle.com/datasets/grubenm/austin-weather): Historical temperature, precipitation, humidity, and windspeed for Austin, Texas.

### 2-1. Load data


```python
# Import modules
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose


# SciKit Learn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
from skforecast.utils import save_forecaster, load_forecaster
```


```python
# Load data
github_url = "https://raw.githubusercontent.com/Dieg0Maciel/time_series_forecast_with_scikitlearn/main/"
data_url = github_url + "austin_weather.csv"
data = pd.read_csv(data_url)

# Since we are working with time series, lets set our index to be the date
data = data.set_index("Date")

# Cast the index as a datetime type
data.index = pd.to_datetime(data.index)

# Load data description
description_url = github_url + "data_description.json"
data_description = requests.get(description_url).json()

# Print data info
data.info()

# Print data description
pd.DataFrame.from_dict(data_description, orient="index")
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1319 entries, 2013-12-21 to 2017-07-31
    Data columns (total 20 columns):
     #   Column                      Non-Null Count  Dtype 
    ---  ------                      --------------  ----- 
     0   TempHighF                   1319 non-null   int64 
     1   TempAvgF                    1319 non-null   int64 
     2   TempLowF                    1319 non-null   int64 
     3   DewPointHighF               1319 non-null   object
     4   DewPointAvgF                1319 non-null   object
     5   DewPointLowF                1319 non-null   object
     6   HumidityHighPercent         1319 non-null   object
     7   HumidityAvgPercent          1319 non-null   object
     8   HumidityLowPercent          1319 non-null   object
     9   SeaLevelPressureHighInches  1319 non-null   object
     10  SeaLevelPressureAvgInches   1319 non-null   object
     11  SeaLevelPressureLowInches   1319 non-null   object
     12  VisibilityHighMiles         1319 non-null   object
     13  VisibilityAvgMiles          1319 non-null   object
     14  VisibilityLowMiles          1319 non-null   object
     15  WindHighMPH                 1319 non-null   object
     16  WindAvgMPH                  1319 non-null   object
     17  WindGustMPH                 1319 non-null   object
     18  PrecipitationSumInches      1319 non-null   object
     19  Events                      1319 non-null   object
    dtypes: int64(3), object(17)
    memory usage: 216.4+ KB





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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Date</th>
      <td>YYYY-MM-DD from 2013-12-21 to 2017-07-31</td>
    </tr>
    <tr>
      <th>TempHighF</th>
      <td>High temperature, in Fahrenheit</td>
    </tr>
    <tr>
      <th>TempAvgF</th>
      <td>Average temperature, in Fahrenheit</td>
    </tr>
    <tr>
      <th>TempLowF</th>
      <td>Low temperature, in Fahrenheit</td>
    </tr>
    <tr>
      <th>DewPointHighF</th>
      <td>High dew point, in Fahrenheit</td>
    </tr>
    <tr>
      <th>DewPointAvgF</th>
      <td>Average dew point, in Fahrenheit</td>
    </tr>
    <tr>
      <th>DewPointLowF</th>
      <td>Low dew point, in Fahrenheit</td>
    </tr>
    <tr>
      <th>HumidityHighPercent</th>
      <td>High humidity, as a percentage</td>
    </tr>
    <tr>
      <th>HumidityAvgPercent</th>
      <td>Average humidity, as a percentage</td>
    </tr>
    <tr>
      <th>HumidityLowPercent</th>
      <td>Low humidity, as a percentage</td>
    </tr>
    <tr>
      <th>SeaLevelPressureHighInches</th>
      <td>High sea level pressure, in inches</td>
    </tr>
    <tr>
      <th>SeaLevelPressureAvgInches</th>
      <td>Average sea level pressure, in inches</td>
    </tr>
    <tr>
      <th>SeaLevelPressureLowInches</th>
      <td>Low sea level pressure, in inches</td>
    </tr>
    <tr>
      <th>VisibilityHighMiles</th>
      <td>High visibility, in miles</td>
    </tr>
    <tr>
      <th>VisibilityAvgMiles</th>
      <td>Average visibility, in miles</td>
    </tr>
    <tr>
      <th>VisibilityLowMiles</th>
      <td>Low visibility, in miles</td>
    </tr>
    <tr>
      <th>WindHighMPH</th>
      <td>High wind speed, in miles per hour</td>
    </tr>
    <tr>
      <th>WindAvgMPH</th>
      <td>Average wind speed, in miles per hour</td>
    </tr>
    <tr>
      <th>WindGustMPH</th>
      <td>Highest wind speed gust, in miles per hour</td>
    </tr>
    <tr>
      <th>PrecipitationSumInches</th>
      <td>Total precipitation, in inches.'T' if Trace</td>
    </tr>
    <tr>
      <th>Events</th>
      <td>Adverse weather events. ' ' if None</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
As we can see: TempAvgF = (TempHighF + TempLowF)/2
"""
temp_comparison = pd.DataFrame()
temp_comparison["TempAvgF"] = data["TempAvgF"]
temp_comparison["(TempHighF + TempLowF)/2"] = (data["TempHighF"]+data["TempLowF"])/2
temp_comparison
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
      <th>TempAvgF</th>
      <th>(TempHighF + TempLowF)/2</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-12-21</th>
      <td>60</td>
      <td>59.5</td>
    </tr>
    <tr>
      <th>2013-12-22</th>
      <td>48</td>
      <td>47.5</td>
    </tr>
    <tr>
      <th>2013-12-23</th>
      <td>45</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>2013-12-24</th>
      <td>46</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>2013-12-25</th>
      <td>50</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-07-27</th>
      <td>89</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>2017-07-28</th>
      <td>91</td>
      <td>90.5</td>
    </tr>
    <tr>
      <th>2017-07-29</th>
      <td>92</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>2017-07-30</th>
      <td>93</td>
      <td>92.5</td>
    </tr>
    <tr>
      <th>2017-07-31</th>
      <td>88</td>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
<p>1319 rows × 2 columns</p>
</div>




```python
"""
We are going to keep only the avg columns
"""

columns_to_drop = [
    'TempHighF', 'TempLowF', 'DewPointHighF', 'DewPointLowF', 
    'HumidityHighPercent', 'HumidityLowPercent', 
    'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches',
    'VisibilityHighMiles',  'VisibilityLowMiles'
]

data.drop(columns_to_drop, axis=1, inplace=True)
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1319 entries, 2013-12-21 to 2017-07-31
    Data columns (total 10 columns):
     #   Column                     Non-Null Count  Dtype 
    ---  ------                     --------------  ----- 
     0   TempAvgF                   1319 non-null   int64 
     1   DewPointAvgF               1319 non-null   object
     2   HumidityAvgPercent         1319 non-null   object
     3   SeaLevelPressureAvgInches  1319 non-null   object
     4   VisibilityAvgMiles         1319 non-null   object
     5   WindHighMPH                1319 non-null   object
     6   WindAvgMPH                 1319 non-null   object
     7   WindGustMPH                1319 non-null   object
     8   PrecipitationSumInches     1319 non-null   object
     9   Events                     1319 non-null   object
    dtypes: int64(1), object(9)
    memory usage: 113.4+ KB



```python
# Data plot
data["TempAvgF"].plot(figsize=(15, 5), title='Avg. Temperature in Farenheit')
plt.show()
```


    
![png](output_9_0.png)
    



```python
data["TempAvgF"].loc[(data.index > '2016-03-01') & (data.index < '2016-10-30')] \
    .plot(figsize=(15, 5), title='Avg. Temperature in Farenheit')
plt.show()
```


    
![png](output_10_0.png)
    


### 2-2. Missing data


```python
"""
PrecipitationSumInches column:
    Using the *.unique()* method to see how the data has been encoded 
    we conclude there are no missing values.The string *'T'* represent 
    traces of precipitation in PrecipitationSumInches column 
"""
data['PrecipitationSumInches'].unique()
```




    array(['0.46', '0', 'T', '0.16', '0.1', '0.01', '0.06', '0.05', '0.02',
           '0.15', '0.11', '0.08', '0.17', '0.74', '0.07', '0.2', '0.27',
           '1.34', '2.45', '0.94', '0.14', '0.19', '1.56', '1.75', '0.55',
           '1.49', '0.24', '0.49', '0.31', '3.53', '1.52', '0.09', '0.98',
           '0.22', '0.51', '0.68', '0.35', '3.66', '0.13', '0.21', '0.03',
           '1.42', '0.3', '0.56', '1.51', '0.04', '0.33', '3.33', '0.59',
           '0.63', '0.76', '2.07', '0.4', '0.26', '0.45', '2.17', '1.17',
           '0.43', '0.41', '0.57', '3.84', '0.29', '0.23', '1.12', '0.36',
           '2.6', '1.41', '5.2', '0.67', '2.66', '1.09', '0.93', '0.75',
           '1.05', '0.79', '4.79', '0.65', '4.93', '0.89', '0.53', '1.03',
           '1.46', '0.25', '1.54', '1.32', '1.33', '2.18', '0.34', '1.19',
           '1.13', '0.58', '0.54', '0.92', '0.77', '2.25', '0.52', '0.18',
           '1.07', '1.61', '1.06', '2.35', '1.79', '1.22', '1.29', '0.37',
           '0.61', '1.57', '0.86', '0.66', '0.73', '0.71', '0.12', '0.47',
           '1.39'], dtype=object)




```python
"""
Event column:
    Using the .unique() method to see how the data has been encoded 
    we conclude there are no missing values. The string ' ' represent 
    no event in the Event column
"""

data["Events"].unique()
```




    array(['Rain , Thunderstorm', ' ', 'Rain', 'Fog', 'Rain , Snow',
           'Fog , Rain', 'Thunderstorm', 'Fog , Rain , Thunderstorm',
           'Fog , Thunderstorm'], dtype=object)




```python
"""
Numerical columns encoded as strings:
    We should be careful using the pandas method .isnull() for numerical columns 
    were the data is encoded as strings. In this particular dataset missing values 
    were encoded as "-". Therefore, counting the number of strings "-" for each column 
    gives us the amount of missing values
"""

cols = data.columns.to_list()
cols_to_exclude = [
    'TempAvgF', 'PrecipitationSumInches', 'Events'
]
cols_to_transfor_to_numeric = [col for col in cols if col not in cols_to_exclude]

for col in cols_to_transfor_to_numeric:
    print(f"{col}: {data[col][data[col] == '-'].count()}")
```

    DewPointAvgF: 7
    HumidityAvgPercent: 2
    SeaLevelPressureAvgInches: 3
    VisibilityAvgMiles: 12
    WindHighMPH: 2
    WindAvgMPH: 2
    WindGustMPH: 4


### 2-3. Casting to numerical


```python
"""
PrecipitationSumInches column:
    Since the categorical value 'T' corresponds to traces, lets separate it 
    from PrecipitationSumInches by replacing 'T' for '0.001' 
"""

# Replace 'T' by '0.001'
precipitation = data.PrecipitationSumInches.to_list()
for i, p in enumerate(precipitation):
    if p == 'T':
        precipitation[i] = '0.001'
data['PrecipitationSumInches'] = precipitation

# Cast PrecipitationSumInches as numeric
data['PrecipitationSumInches'] = pd.to_numeric(data['PrecipitationSumInches'], errors='coerce')
```


```python
"""
Numerical columns encoded as strings
"""

for col in cols_to_transfor_to_numeric:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
# Use the method .isnull() to check if errors='coerce' worked properly
data[cols_to_transfor_to_numeric].isnull().sum()
```




    DewPointAvgF                  7
    HumidityAvgPercent            2
    SeaLevelPressureAvgInches     3
    VisibilityAvgMiles           12
    WindHighMPH                   2
    WindAvgMPH                    2
    WindGustMPH                   4
    dtype: int64



### 2-4. Adding features


```python
"""
Event colum:
    Create a feature for each event and drop the Event column
"""

rain = ['Rain','Fog , Rain', 'Rain , Thunderstorm', 'Rain , Snow', 'Fog , Rain , Thunderstorm']
fog = ['Fog','Fog , Rain', 'Fog , Thunderstorm', 'Fog , Rain , Thunderstorm']
snow = ['Snow', 'Rain , Snow']
thunderstorm = ['Thunderstorm', 'Rain , Thunderstorm', 'Fog , Thunderstorm', 'Fog , Rain , Thunderstorm']

# Add a new feature for each event
data["Rain"] = [event in rain for event in data.Events.to_list()]
data["Fog"] = [event in fog for event in data.Events.to_list()]
data["Snow"] = [event in snow for event in data.Events.to_list()]
data["Thunderstorm"] = [event in thunderstorm for event in data.Events.to_list()]

# Drop Events column
data.drop('Events', inplace=True, axis=1)
```

### 2-5. Time series components


```python
period = 365 # in days

# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(
    data["TempAvgF"], model="multiplicative", period=period
)

# Additive Decomposition 
additive_decomposition = seasonal_decompose(
    data["TempAvgF"], model="additive", period=period
)

#Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
```


    
![png](output_21_0.png)
    



    
![png](output_21_1.png)
    



```python
"""
Autocorrelation Function (ACF) plot.
"""
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
autocorrelation_plot(data["TempAvgF"].tolist())
plt.show()
```


    
![png](output_22_0.png)
    



```python
"""
Lag Plots
"""
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(data["TempAvgF"], lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Avg. Temperature', y=1.05)    
plt.show()
```


    
![png](output_23_0.png)
    


### 2-6. Linear correlations: Heatmap and Scatterplots


```python
"""
Heatmap
"""
sns.heatmap(data.corr(), annot=True)
plt.show()
```


    
![png](output_25_0.png)
    


#### Scatterplots


```python
numeric_cols = [
    'TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 
    'SeaLevelPressureAvgInches', 'VisibilityAvgMiles'
]

g = sns.PairGrid(data[numeric_cols])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
```




    <seaborn.axisgrid.PairGrid at 0x7f0d72f419e8>




    
![png](output_27_1.png)
    



```python
numeric_cols = [
    'TempAvgF',  'WindHighMPH', 'WindAvgMPH', 'WindGustMPH', 'PrecipitationSumInches'
]

g = sns.PairGrid(data[numeric_cols])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
```




    <seaborn.axisgrid.PairGrid at 0x7f0d7172e668>




    
![png](output_28_1.png)
    


## 3. MODEL SELECTION AND TRAINING


```python
"""Train-Test Split"""
label = 'TempAvgF'
features = data.columns.tolist()
features.remove(label)

X = data[features]
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=False
)

# Plot train-test split
fig, ax = plt.subplots(figsize=(15, 5))
y_train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
y_test.plot(ax=ax, label='Test Set')
ax.axvline(y_test.index[0], color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()
```


    
![png](output_30_0.png)
    



```python
""" Model"""
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)

"""Training"""
reg.fit(X_train, y_train)
```

    [12:40:42] WARNING: ../src/objective/regression_obj.cu:188: reg:linear is now deprecated in favor of reg:squarederror.
    [12:40:42] WARNING: ../src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, early_stopping_rounds=50,
                 enable_categorical=False, gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.01, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=nan,
                 monotone_constraints='()', n_estimators=1000, n_jobs=4,
                 num_parallel_tree=1, objective='reg:linear', predictor='auto',
                 random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 subsample=1, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
"""Feature Importance"""
feat_imp = reg.get_booster().get_score(importance_type='gain')
fi = pd.DataFrame({'Features':list(feat_imp.keys()), 'Importance':list(feat_imp.values())})
fi = fi.set_index('Features')
fi.sort_values('Importance').plot(kind='barh', title='Feature Importance')
plt.show()
fi.sort_values(by='Importance', ascending=False)
```


    
![png](output_32_0.png)
    





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
      <th>Importance</th>
    </tr>
    <tr>
      <th>Features</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DewPointAvgF</th>
      <td>3289.620117</td>
    </tr>
    <tr>
      <th>HumidityAvgPercent</th>
      <td>435.336914</td>
    </tr>
    <tr>
      <th>Rain</th>
      <td>254.712479</td>
    </tr>
    <tr>
      <th>SeaLevelPressureAvgInches</th>
      <td>223.630096</td>
    </tr>
    <tr>
      <th>PrecipitationSumInches</th>
      <td>196.618988</td>
    </tr>
    <tr>
      <th>VisibilityAvgMiles</th>
      <td>135.365005</td>
    </tr>
    <tr>
      <th>WindHighMPH</th>
      <td>30.398726</td>
    </tr>
    <tr>
      <th>WindGustMPH</th>
      <td>20.418079</td>
    </tr>
    <tr>
      <th>Fog</th>
      <td>15.400835</td>
    </tr>
    <tr>
      <th>WindAvgMPH</th>
      <td>12.347539</td>
    </tr>
  </tbody>
</table>
</div>



## 4. EVALUATION AND PREDICTIONS 


```python
"""Predictions"""
predictions = reg.predict(X_test)
y_pred = pd.DataFrame(data=predictions, index=y_test.index, columns=['Predictions'])

score = mean_absolute_error(y_test.values, y_pred.values)
print(f'MAE Score on the Test set: {score:0.2f}')

fig, ax = plt.subplots(figsize=(15, 5))
y_test.plot(ax=ax, label='Test Set')
y_pred.plot(ax=ax, label='Predictions')
ax.legend(['Test Set', 'Predictions'])
plt.show()
```

    MAE Score on the Test set: 1.72



    
![png](output_34_1.png)
    


## 5. CONCLUSION 

## 6. FUTURE WORK

## 7. REFERENCES

* [Temperature prediction time series](https://www.kaggle.com/code/tudorpreduna/temperature-pred-time-series)
* [Time Series Forecasting with XGBoost](https://www.youtube.com/watch?v=vV12dGe_Fho)
* [Finding Seasonal Trends in Time-Series Data with Python](https://towardsdatascience.com/finding-seasonal-trends-in-time-series-data-with-python-ce10c37aa861)
* [Complete Guide on Time Series Analysis in Python](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python)
* [Time Series as Features](https://www.kaggle.com/code/ryanholbrook/time-series-as-features)
* [Skforecast: time series forecasting with Python and Scikit-learn](https://cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html)
* [How to do Time Series Split using Sklearn](https://medium.com/@Stan_DS/timeseries-split-with-sklearn-tips-8162c83612b9)


```python

```
