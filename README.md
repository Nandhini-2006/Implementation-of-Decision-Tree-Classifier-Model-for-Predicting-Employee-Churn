# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: NANDHINI N

RegisterNumber: 212224040212

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt. predict(x_test)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

## Output:

![Screenshot 2025-05-19 153755](https://github.com/user-attachments/assets/6545392c-deb8-41f7-9152-6021726bd445)

![Screenshot 2025-05-19 153805](https://github.com/user-attachments/assets/1f9f4df0-c0ce-4845-903d-5821be12643a)

![Screenshot 2025-05-19 153813](https://github.com/user-attachments/assets/6575d7f7-7f4c-4960-aaa2-7bfad6f5a95f)

![Screenshot 2025-05-19 153823](https://github.com/user-attachments/assets/42249eff-4ba5-495b-ad00-00d15e8ea38a)

![Screenshot 2025-05-19 153838](https://github.com/user-attachments/assets/e6825cd6-21cf-432d-95e7-93d09cc4b3bc)

![Screenshot 2025-05-19 153849](https://github.com/user-attachments/assets/54c5dfc6-a4f9-4e51-b31b-b1a5aac59b80)

![Screenshot 2025-05-19 153855](https://github.com/user-attachments/assets/d52c22a0-4995-4da6-a680-255759321912)

![Screenshot 2025-05-19 153904](https://github.com/user-attachments/assets/84b958d6-2410-4e49-a1ce-70d2b06d7b47)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
