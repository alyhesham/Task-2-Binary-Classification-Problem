import pandas as pd
from keras import optimizers
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC  # "Support Vector Classifier"
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
import random
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import LSTM
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV

random.seed(2)

def encode_and_bind(original_dataframe, feature_to_encode):
  dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], drop_first=True, dummy_na=True)
  res = pd.concat([original_dataframe, dummies], axis=1)
  res = res.drop([feature_to_encode], axis=1)
  return res


path = "/Users/Aly/Desktop/binary_classifier_data/"

train = pd.read_csv(path + "training_new.csv", na_filter=False)
validation = pd.read_csv(path + "validation_new.csv", na_values="NA")


train["classLabel"].replace({"no.": 0, "yes.": 1}, inplace=True)
validation["classLabel"].replace({"no.": 0, "yes.": 1}, inplace=True)
train = train.drop(['variable18'], axis=1)
validation = validation.drop(['variable18'], axis=1)
train = train.drop(['variable19'], axis=1)
validation = validation.drop(['variable19'], axis=1)
x = train.corr()
z = validation.corr()


for index, row in train.iterrows():
  if row["classLabel"] == 1:
    for col in train.columns:
       if row[col] == "NA":
         train.drop(index, inplace=True)
         break


train.replace('NA', np.nan, inplace=True)

validation.replace('NA', np.nan, inplace=True)

data_no= train[train['classLabel'] == 0]
data_yes= train[train['classLabel'] == 1]

data_no = data_no.apply(lambda x: x.fillna(x.value_counts().index[0]))
validation = validation.apply(lambda x: x.fillna(x.value_counts().index[0]))


all = pd.concat([data_no,data_yes])

train = all

y_train = train.classLabel
y_test = validation.classLabel

print(Counter(y_train))

train = train.drop(["classLabel"], axis=1)
validation = validation.drop(["classLabel"], axis=1)

over = RandomOverSampler(sampling_strategy=0.13)

train, y_train = over.fit_resample(train, y_train)
print(Counter(y_train))

under = RandomUnderSampler(sampling_strategy=0.8)
train, y_train = under.fit_resample(train, y_train)

print(Counter(y_train))
limit = len(train)


newdf = pd.concat([train, validation])
newdf.fillna(newdf.median(), inplace=True)

features_to_encode = ['variable1', 'variable4', 'variable5',
                      'variable6', 'variable7', 'variable9', 'variable10', 'variable12', 'variable13']


for feature in features_to_encode:
  newdf = encode_and_bind(newdf, feature)


cols = newdf.columns

min_max_scaler = preprocessing.MinMaxScaler()
newdf = min_max_scaler.fit_transform(newdf)

train = newdf[:limit, :]
validation = newdf[limit:, :]

train = pd.DataFrame(train, columns= cols)

validation = pd.DataFrame(validation, columns= cols)

classifier = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=54, max_depth=8)
#RandomForestClassifier(n_estimators=20, random_state=0, max_depth=2)
classifier.fit(train, y_train)
y_pred = classifier.predict(validation)
y_pred = pd.Series(y_pred)
print("Accuracy : ", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred)
print(report)
cm = confusion_matrix(y_test, y_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(cm.diagonal())




'''

RandomForestClassifier(n_estimators=200, max_features='auto', random_state=54, max_depth=8)

classifier = SVC(C=100, gamma=1, kernel='rbf', verbose=False)

classifier = GaussianNB()
classifier.fit(train, y_train)
y_pred = classifier.predict(validation)
y_pred = pd.Series(y_pred)
print("Accuracy : ", accuracy_score(y_test, y_pred))


classifier = LogisticRegression(random_state =0, multi_class="ovr", solver= "liblinear", max_iter= 300)

RandomForestClassifier(n_estimators=100, random_state=42, max_depth=2)

KNeighborsClassifier(n_neighbors=80)

model = Sequential()

model.add(Dense(36, input_dim=47, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=optimizers.Adam(lr=0.002), loss="binary_crossentropy", metrics=['accuracy'])

model.fit(train, y_train, epochs=60, batch_size=64, validation_data=[validation,y_test])

test_loss, test_acc = model.evaluate(validation, y_test)
print('Test accuracy:', test_acc)





param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(train, y_train)

print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


param_grid = {
    'n_estimators': [10,20,30,50,100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(train, y_train)

print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)







newdf = pd.concat([train, validation])

features_to_encode = ['variable1', 'variable4', 'variable5',
                      'variable6', 'variable7', 'variable9', 'variable10', 'variable12', 'variable13', 'variable18']

for feature in features_to_encode:
  newdf = encode_and_bind(newdf, feature)

# for feature in features_to_encode:
#   validation = encode_and_bind(validation, feature)
cols = newdf.columns

newdf.fillna(newdf.median(), inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()
newdf = min_max_scaler.fit_transform(newdf)

train = newdf[:limit, :]
validation = newdf[limit:, :]

train = pd.DataFrame(train, columns= cols)
validation = pd.DataFrame(validation, columns= cols)


classifier = SVC(kernel="poly")
#RandomForestClassifier(n_estimators=20, random_state=0, max_depth=2)
classifier.fit(train, y_train)
y_pred = classifier.predict(validation)
y_pred = pd.Series(y_pred)
print("Accuracy : ", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred)
print(report)
cm = confusion_matrix(y_test, y_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(cm.diagonal())
'''
