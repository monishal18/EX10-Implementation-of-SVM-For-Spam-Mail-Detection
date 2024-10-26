# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Convert emails into numerical features using tokenization, lowercasing, and TF-IDF or Bag
of Words.
2. Transform the processed text into feature vectors for SVM input.
3. Train an SVM classifier (with a linear or other kernel) on labeled data to distinguish
between spam and not spam emails.
4. Use the trained SVM model to predict whether new emails are spam and evaluate
performance using metrics like accuracy and precision.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: monisha.L
RegisterNumber:  2305001019
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv("/content/spamEX10.csv",encoding='ISO-8859-1')
df.head()
v=CountVectorizer()
x=v.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train, y_train)
p=model.predict(x_test)
print("ACCURACY:",accuracy_score(y_test,p))
print("Classification Report:")
print(classification_report(y_test,p))
def predict_message(message):
message_vec=v.transform([message])
e=model.predict(message_vec)
return e[0]
n="Congratulations!"
result=predict_message(n)
print(f"The message:'{n}' is classified as:{result}")

```

## Output:
![image](https://github.com/user-attachments/assets/d69f80f3-1c4c-4a79-9aff-712ee640e584)
![image](https://github.com/user-attachments/assets/3257fc1c-af2c-492a-bd20-70df2344f273)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
