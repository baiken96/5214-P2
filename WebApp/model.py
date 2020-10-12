import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv')
data.head(3)

data['date'] = data['date'].apply(pd.to_datetime)
data['year'] = [i.year for i in data['date']]
data['month'] = [i.month_name()[0:3] for i in data['date']]
data['day'] = [i.day_name()[0:3] for i in data['date']]

dat = data[['temperature', 'humidity', 'windspeed']]
labels = data['count']

train_data, test_data, train_labels, test_labels = train_test_split(dat, labels, test_size=0.1, random_state=1)

classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)

classifier.fit(train_data, train_labels, verbose=1)

with open('model/bike_model_xgboost.pkl', 'wb+') as file:
    pickle.dump(classifier, file)