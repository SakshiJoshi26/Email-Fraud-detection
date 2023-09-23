import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/mail_data.csv')

print(df)

data = df.where((pd.notnull(df)), " ")
data.head(10)

data.info()
data.shape
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

X = data['Message']

Y = data['Category']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state =3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)
print(X_train_features)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data )

print("Accuracy on the training dataset:",accuracy_on_training_data)

prediction_on_testing_data = model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)

print("Accuracy on the training dataset:",accuracy_on_testing_data)

input_mail = ["Your Application Has Been Sent! Dear Sakshi Joshi,You have submitted an application for the following using firstnaukri.com"]
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)

#print(prediction)

if(prediction[0]==1):
  print("This is a Ham mail")
else:
  print("This is a spam mail")