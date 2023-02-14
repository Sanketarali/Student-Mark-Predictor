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

