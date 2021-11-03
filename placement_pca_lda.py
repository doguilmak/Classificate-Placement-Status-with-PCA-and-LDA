# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:49:07 2021

@author: doguilmak

dataset: https://www.kaggle.com/benroshan/factors-affecting-campus-placement

"""
#%%
#  1. Libraries

import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('Placement_Data_Full_Class.csv')
print(df.head())
print(df.info())

# 2.2. Removing Unnecessary Columns
df.drop(['sl_no'], axis = 1, inplace = True)

# 2.3. Plot Gender and Status on Pie Chart
explode = (0, 0.05)
fig = plt.figure(figsize = (12, 12), facecolor='w')
out_df=pd.DataFrame(df.groupby('gender')['gender'].count())

patches, texts, autotexts = plt.pie(out_df['gender'], autopct='%1.1f%%',
                                    textprops={'color': "w"},
                                    explode=explode,
                                    startangle=90, shadow=True)

for patch in patches:
    patch.set_path_effects({path_effects.Stroke(linewidth=2.5,
                                                foreground='w')})

plt.legend(labels=['Female','Male'], bbox_to_anchor=(1., .95), title="Gender")
plt.show()


explode = (0, 0.05)
fig = plt.figure(figsize = (12, 12), facecolor='w')
out_df=pd.DataFrame(df.groupby('status')['status'].count())

patches, texts, autotexts = plt.pie(out_df['status'], autopct='%1.1f%%',
                                    textprops={'color': "w"},
                                    colors=['#228B22','#1DACD6'],
                                    explode=explode,
                                    startangle=90, shadow=True)

for patch in patches:
    patch.set_path_effects({path_effects.Stroke(linewidth=2.5,
                                                foreground='w')})

plt.legend(labels=['Unplaced','Placed'], bbox_to_anchor=(1., .95), title="Status")
plt.show()
   
# 2.4. Looking For Anomalies
print("{} duplicated.".format(df.duplicated().sum()))
dp = df[df.duplicated(keep=False)]
dp.head(2)
df.drop_duplicates(inplace= True)
print("{} duplicated.".format(df.duplicated().sum()))
print(df.describe().T)

# 2.5. Label Encoding 
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
df['salary'] = df['salary'].fillna(0)
print("data:\n", df)

# 2.6. Seperate the Data Depending on Dependent and Independent Variableles
y = df["status"]
X = df.drop("status", axis = 1)

# 2.7. Split as Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# 2.8. Scaling Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# 3.1. PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  # 2 dimensional

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

principalDf = pd.DataFrame(data = X_train2,
              columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['status']]], axis = 1)

fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(1, 1, 1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['status'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 20)
ax.legend(['Non Placed', 'Placed'])
ax.grid()

# 3.2. LR Transform Before PCA
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 3.3. LR After PCA Transform
classifier2 = LogisticRegression(random_state=0, C=2, tol=1)
classifier2.fit(X_train2, y_train)

# 3.4. Predictions
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 3.5. Actual / Without PCA 
print('Actual / Without PCA')
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}\n")

# 3.6. Actual / Result after PCA
print("Actual / With PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
print(f"Accuracy score: {accuracy_score(y_test, y_pred2)}\n")

# 3.7. After PCA / Before PCA
print('Without PCA and with PCA')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)
print(f"Accuracy score: {accuracy_score(y_pred, y_pred2)}\n")

#%%
# 4. LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)  # 1 dimensional

X_train_lda = lda.fit_transform(X_train, y_train)  # In order for LDA to learn, the y_train parameter is entered.
X_test_lda = lda.transform(X_test)

# 4.1. After LDA Transform
classifier_lda = LogisticRegression(random_state=0, tol=1, C=1)
classifier_lda.fit(X_train_lda, y_train)

# 4.2. Predict LDA Datas
y_pred_lda = classifier_lda.predict(X_test_lda)

# 4.3. After LDA / Actual
print('LDA and Actual')
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)
print(f"Accuracy score: {accuracy_score(y_pred, y_pred_lda)}\n")

#%%
# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = classifier_lda, 
                          X=X_train_lda, 
                          y=y_train, 
                          cv = 4)

print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())


# Grid Search
from sklearn.model_selection import GridSearchCV
p = [{'tol':[1e-4,1e-3,1e-2,1e-1,1], 'C':[1,2,3,4,5], 
      'multi_class':['auto', 'ovr', 'multinomial']},
     {'tol':[1e-4,1e-3,1e-2,1e-1,1], 'C':[1,2,3,4,5], 
      'multi_class':['auto', 'ovr', 'multinomial']},
     {'tol':[1e-4,1e-3,1e-2,1e-1,1], 'C':[1,2,3,4,5], 
      'multi_class':['auto', 'ovr', 'multinomial']},
     {'tol':[1e-4,1e-3,1e-2,1e-1,1], 'C':[1,2,3,4,5], 
      'multi_class':['auto', 'ovr', 'multinomial']}]


gs = GridSearchCV(estimator= classifier_lda,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search:")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
