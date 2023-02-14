# Student-Mark-Predictor
This project is a machine learning model that predicts the marks of a student based on number of courses and time of studied performance. The model is trained on a dataset of historical student data, and can be used to predict the marks of students.

# Prerequisites
To run this project, you will need the following:<br>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

# Student Marks Prediction (Case Study)
You are given some information about students like:<br>

1.the number of courses they have opted for<br>
2.the average time studied per day by students<br>
3.marks obtained by students<br>

# How  did I do?

The dataset I am using for the student marks prediction task is downloaded from Kaggle. Now let’s start with this task by importing the necessary Python libraries and dataset:<br>

import numpy as np<br>
import pandas as pd<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import LinearRegression<br>

data = pd.read_csv("Student_Marks.csv")<br>

Now before moving forward, let’s have a look at whether this dataset contains any null values or not:<br>

print(data.isnull().sum())<br>

The dataset is ready to use because there are no null values in the data. There is a column in the data containing information about the number of courses students have chosen. Let’s look at the number of values of all values of this column:<br>

data["number_courses"].value_counts()<br>

# Student Marks Prediction Model
Now let’s move to the task of training a machine learning model for predicting the marks of a student. Here, I will first start by splitting the data into training and test sets:<br>

X = data.drop(['Marks'],axis=1)<br>
y=data['Marks']<br>
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)<br>

Now I will train a machine learning model using the linear regression algorithm:<br>

lr = LinearRegression()<br>
lr.fit(X_train,y_train)<br>

y_pred1 = lr.predict(X_test)<br>
score1 = metrics.r2_score(y_test,y_pred1)<br>

score1<br>

0.9459936100591214<br>

# Save the Model

import joblib<br>
joblib.dump(rf,'model_joblib_test')<br>
model = joblib.load('model_joblib_test')<br>

Now, Lets check the output with a sample input<br>

model.predict([[3,4.5]])<br>

array([20.11396])<br>
# Conclusion
So this is how you can predict the marks of a student with machine learning using Python.<br>
