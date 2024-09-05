# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHALINI VENKATESULU
RegisterNumber:212223220104
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


## Output:
df.head()

![image](https://github.com/user-attachments/assets/136efd4d-32b1-4286-89aa-6233ee7fada4)

df.tail()

![image](https://github.com/user-attachments/assets/c610bcd7-c415-4f40-88ea-80ac8f3a7cca

Array value of X

![image](https://github.com/user-attachments/assets/1cb8b47f-21c1-48b8-afec-c3906a7cdf23)

Array value of Y

![image](https://github.com/user-attachments/assets/1c3e72d9-0c26-4d27-b5c4-8540ec0278de)

Value of Y prediction

![image](https://github.com/user-attachments/assets/13613731-f9e9-44de-807c-c03bbe493e92

Array value of Y test

![image](https://github.com/user-attachments/assets/2cb466bb-d29c-4a9c-99ed-111c52f9f118)

Training Set Graph

![image](https://github.com/user-attachments/assets/7c783034-ca39-4841-9219-f70292667760)

Test Set Graph

![image](https://github.com/user-attachments/assets/56eb95fd-7fe7-4b9e-9757-96dbabf7463e)

Values of MSE,MAE and RMSE


![image](https://github.com/user-attachments/assets/da22c2e0-2795-46a2-aee8-73b304826e0f)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
