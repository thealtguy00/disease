from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
#from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
df_train=pd.read_csv('training_data.csv')
# dis=[]
# for x in range(len(df)):
#     if df.iloc[x]['prognosis'] not in dis:
#         dis.append(df.iloc[x]['prognosis'])
# print(dis)

# import random
# for x in range(10):
#     print(random.randint(1,5))
import random

cols = df_train.columns
cols = cols[:-2]
train_features = df_train[cols]
age=[]
for x in range(len(df_train)):
    if df_train.iloc[x]['prognosis']=='Diabetes':
        age.append(random.randint(40,67))
    elif df_train.iloc[x]['prognosis']=='Gastroenteritis':
        age.append(random.randint(0,7))
    elif df_train.iloc[x]['prognosis']=='Bronchial Asthma':
        age.append(random.randint(0,7))
    elif df_train.iloc[x]['prognosis'].rstrip()=='hypertension':
        age.append(random.randint(50,70))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Cervical spondylosis':
        age.append(random.randint(40,60))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Paralysis (brain hemorrhage)':
        age.append(random.randint(60,100))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Dimorphic hemmorhoids(piles)':
        age.append(random.randint(40,75))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Alcoholic hepatitis':
        age.append(random.randint(20,85))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Heart attack)':
        age.append(random.randint(40,85))
    elif df_train.iloc[x]['prognosis'].rstrip()=='Osteoarthristis':
        age.append(random.randint(50,90))
    else:
        age.append(random.randint(1,100))
df_train['age']=age
print(df_train)
cols = df_train.columns
colss = [col for col in cols]
colss = colss[:-2]
colss.append('age')
colss.append(colss.pop(-2))
#print(colss)
train_features = df_train[colss]
train_labels = df_train['prognosis']




X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels)
classifier = RandomForestClassifier()
classifier.fit(X=X_train, y=y_train)
confidence = classifier.score(X_val, y_val)
# Validation Data Prediction
y_pred = classifier.predict(X_val)
# Model Validation Accuracy
accuracy = accuracy_score(y_val, y_pred)
print(accuracy)

# Check for data sanity
# assert (len(test_features.iloc[0]) == 132)
# assert (len(test_labels) == test_features.shape[0])