import pandas as pd
import numpy as np  
import seaborn as sns 
#Reading csv file
url = "http://bit.ly/w-data"
score_data=pd.read_csv(url)


#creating copy
sd=score_data.copy()


# structure of dataset
sd.info()
print(sd)


#removing duplicate records sd.drop_duplicate(keep='first',inplace=true)

#Data cleaning
sd=sd.dropna(axis=0)


#plotting
sns.regplot(x='Hours',y='Scores',scatter=True,fit_reg=False,data =sd)

#importing Necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Model Building

#seperating input and output features
x1=sd.iloc[:, :-1].values 
y1=sd.iloc[:, 1].values


#splitting data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(x1,y1,test_size=.3,random_state=0)


#Linear Regression
lgr=LinearRegression(fit_intercept=True)
model_lin1=lgr.fit(X_train,Y_train)
prediction=lgr.predict(X_test)

#Comuting MSE (Model Evaluation)
mse1=mean_squared_error(Y_test,prediction)
rmse=np.sqrt(mse1)
print("Mean Squared Error-  ",rmse)

#Regression Dignostics
residuals=Y_test-prediction
print("Residuals - ",residuals)

#print the prediction
Hours=[[9.25],[9.26]]
pred=lgr.predict(Hours)
print("No of Hours = {}".format(Hours[0]))
print("Predicted Score = {}".format(pred[0]))


#plotting
sns.lmplot(x="Hours",y="Scores",data=sd)