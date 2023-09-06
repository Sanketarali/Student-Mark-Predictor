
# Student-Mark-Predictor
<h3>This project is a machine learning model that predicts the marks of a student based on number of courses and time of studied performance. The model is trained on a dataset of historical student data, and can be used to predict the marks of students.</h3>

# Prerequisites
<h3>To run this project, you will need the following:<br></h3>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

# Student Marks Prediction (Case Study)
 given some information about students like:<br>

1.the number of courses they have opted for<br>
2.the average time studied per day by students<br>
3.marks obtained by students<br>

# How  did I do?

<h3>The dataset I am using for the student marks prediction task is downloaded from Kaggle. Now let’s start with this task by importing the necessary Python libraries and dataset:<br></h3>

import numpy as np<br>
import pandas as pd<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import LinearRegression<br>

data = pd.read_csv("Student_Marks.csv")<br>

data.head()
![1st](https://github.com/Sanketarali/Student-Mark-Predictor/assets/110754364/f260b3c3-0ac3-49b9-b62f-78a0f4a29e72)



<h3>Now before moving forward, let’s have a look at whether this dataset contains any null values or not:<br></h3>

data.isnull().sum()<br>

![2nd](https://github.com/Sanketarali/Student-Mark-Predictor/assets/110754364/9091a564-b0fb-4fa0-88ff-a152da8c3d74)



<h3>The dataset is ready to use because there are no null values in the data. There is a column in the data containing information about the number of courses students have chosen. Let’s look at the number of values of all values of this column:<br></h3>

data["number_courses"].value_counts()<br>

![3rd](https://github.com/Sanketarali/Student-Mark-Predictor/assets/110754364/5f4be19e-19d2-4931-9a34-c5113c301c4f)


# Student Marks Prediction Model
<h3>Now let’s move to the task of training a machine learning model for predicting the marks of a student. Here, I will first start by splitting the data into training and test sets:<br></h3>

X = data.drop(['Marks'],axis=1)<br>
y=data['Marks']<br>
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)<br>

<h3>Now I will train a machine learning model using the linear regression algorithm:<br></h3>

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

<h3>Now, Lets check the output with a sample input<br></h3>

model.predict([[3,4.5]])<br>

array([20.11396])<br>

# Result
![4th](https://github.com/Sanketarali/Student-Mark-Predictor/assets/110754364/87b32920-a13e-4a71-854d-f6c1b44da568)


# Conclusion
<h3>So this is how I predicted the marks of a student with machine learning using Python.<br></h3>

