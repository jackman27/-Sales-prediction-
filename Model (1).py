import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data = pd.read_csv('train.csv')

data.shape
data.head()
data.info()
data.describe()
data.describe(include = 'O')
data.dtypes



data.StateHoliday = data.StateHoliday.astype(str)

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')



store_info = data[data.Store==100].sort_values('Date')
plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_info.Sales.values[:365])

plt.figure(figsize=(20, 10))
plt.scatter(x=store_info[data.Open==1].Promo, y=store_info[data.Open==1].Sales, alpha=0.1)


#check the kurt values to handle outliers

data.kurt().sort_values(ascending=False)

#handling outliers

''''from feature_engine.outlier_removers import Winsorizer


windsoriser = Winsorizer(distribution='gaussian', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                        variables=['Customers'])
windsoriser.fit(data)



data1 = windsoriser.transform(data)'''''

#checking kurt values again after handling outliers

data.kurt().sort_values(ascending=False)

#checking variance

data.var()

#checking correlation between independent variables

corr = data.corr()
plt.figure(figsize = (18,8))
sns.heatmap(corr)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
data3 = data.drop(['Date', 'StateHoliday'], axis =1)
selected_columns = data3.columns[columns]
data4 = data3[selected_columns]

#checking correlation between dependent and independent variables

cor_target = abs(corr["Sales"])
relevant_features = cor_target[cor_target>0.2]

#Removing the unnecessary columns

data_updated = data.drop(['Store', 'Date' ], axis=1)

data_updated = pd.get_dummies(data_updated, columns=['DayOfWeek', 'StateHoliday'])

X = data_updated.drop(['Sales'], axis=1).values
y = data_updated.Sales.values



from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


regressor = DecisionTreeRegressor()
kfolds = KFold(n_splits=4, shuffle=True, random_state=42)
kfolds.get_n_splits(X)
scores = cross_val_score(regressor, X, y, cv=kfolds)
print(cross_val_score(regressor, X, y, cv=kfolds))





X_train = pd.get_dummies(data[data.Store!=100], columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date'], axis=1).values
y_train = pd.get_dummies(data[data.Store!=100], columns=['DayOfWeek', 'StateHoliday']).Sales.values
regressor.fit(X_train, y_train)

y_predict = regressor.predict(pd.get_dummies(store_info, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date'], axis=1).values)


y_result = regressor.predict(pd.get_dummies(data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date'], axis=1).values)

sample = pd.read_csv('sample_submission.csv')
submission_df = pd.DataFrame({'Id': sample.Id, 'Sales': y_result[:41088]})

submission_df.to_csv('Final_submission_file.csv', index = False)

plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_info.Sales.values[:365], label="actual value")
plt.plot(y_predict[:365], c='r', label="predicted value")
plt.legend()

from sklearn import metrics
print(metrics.r2_score(store_info.Sales.values, y_predict))

y_result = regressor.predict(pd.get_dummies(data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date'], axis=1).values)

from sklearn import metrics
print(metrics.r2_score(y, y_result))

#now checking with DL

import keras
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(4, input_dim = 15, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.summary()



model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

epochs_hist = model.fit(X_train, y_train, epochs = 10, batch_size = 10)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])

plt.title('Model Loss Progress During Traininig')
plt.ylabel('Training and validation loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss','Validation Loss'])



predict_y = model.predict(pd.get_dummies(data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date'], axis=1).values)

from sklearn import metrics
print(metrics.r2_score(y, predict_y))


#using Decision tree because better accuracy


from sklearn import metrics
print(metrics.r2_score(y, y_result))



sample = pd.read_csv('sample_submission.csv')
submission_df = pd.DataFrame({'Id': sample.Id, 'Sales': y_result[:41088]})

submission_df.to_csv('Final_submission_file.csv', index = False)

plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_info.Sales.values[:365], label="actual value")
plt.plot(y_predict[:365], c='r', label="predicted value")
plt.legend()
