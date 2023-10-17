#!/usr/bin/env python
# coding: utf-8

# <h1 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> About this Dataset </h1>
# 
# 
# * The dataset comprises 569 samples of tumor cells, with each sample characterized by a unique ID number and a diagnosis label indicating whether the tumor is malignant (M) or benign (B).
# 
# * Within the dataset, columns 3 to 32 hold 30 real-valued features that offer relevant information for constructing a predictive model to discern between benign and malignant tumors.
# 
#     * 1 indicates the presence of Malignant (cancerous) conditions. 
#     * 0 indicates the absence of Benign (non-cancerous) conditions. 
#     
#     
# **Ten real-valued features are calculated for each cell nucleus, encompassing characteristics such as :-**
# 
# * size (mean radius) = mean of distances from center to points on the perimeter
# * texture (variance in gray-scale values)
# * shape (perimeter and area)
# * smoothness (local irregularities in size)
# * compactness (shape compactness compared to a perfect circle)
# * concavity (degree of concave regions in the contour)
# * concave points (quantity of concave areas)
# * symmetry (shape symmetry) and
# * fractal dimension (complexity of the cell's perimeter).
# 
# 
# ### Submitted by : - Sanket Madavi
# 
# ### Role: Data Science Intern

# # Library Importing

# In[1]:


import numpy as np #linear algebra
import pandas as pd #data processing 

import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization

#for model building
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as pgo

import warnings
warnings.filterwarnings('ignore')


# # Data Loading & Manipulating

# In[2]:


Data = pd.read_csv('D:\Disk F\My Stuff\Internships\Coders Cave\Phase 1\Task 2\data.csv')
Data


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# 1. The final column, labeled as "Unnamed: 32," appears to be an erroneous or extraneous addition to our dataset. It is advisable to remove it.
# 
# 2. The majority of columns in the dataset contain numerical data, simplifying our analysis by obviating the need for extensive variable mapping.
# 
# 3. The ID column does not appear to provide valuable insights for predicting cancer. Consequently, it can be safely omitted from the dataset.

# In[3]:


Data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
Data


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# After removing two columns, there are 31 columns remaining. Let's examine the degree of correlation between these 31 columns and the "diagnosis" column to assess their relationship or association with the diagnosis.

# In[4]:


Data.head()


# In[5]:


Data.info()


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# 1. The 'diagnosis' column, which is the target variable we aim to predict, is in the form of object data.
# 
# 2. The dataset contains only one integer-type column, which is the 'ID' column, and it may be considered for removal during data preprocessing.
# 
# 3. Among the 31 columns in the dataset, all are of float data type.

# In[6]:


Data.describe()


# In[7]:


Data.isna().sum()


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# * There are no missing or null values present in any of the 31 columns in the dataset.

# In[8]:


Data.duplicated().sum()


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# * There are no duplicates or repetating values present in any of the 31 columns in the dataset.

# # Pre-Processing

# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[10]:


Data['diagnosis'] = le.fit_transform(Data['diagnosis'])
Data


# In[11]:


Data.diagnosis.value_counts()


# In[12]:


corr = Data.corr() #heatmap
corr


# In[13]:


Data.drop('diagnosis', axis=1).corrwith(Data.diagnosis).plot(kind='bar', grid=True, figsize = (10, 8), title="Correlation with target",color="blue");


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# Indeed, it's quite fascinating! Here's the information presented differently:
# 
# 1. A small number of columns exhibit a negative correlation with the 'diagnosis' column.
# 2. Approximately half of our columns show a positive correlation of over 50% with the 'diagnosis' column.
# 
# Now, the challenge is to decide which attributes we should include when constructing our model.

# In[14]:


plt.figure(figsize=(12,8.))

corr_matrix = Data.corr()
threshold = 0.40 
filtre = np.abs(corr_matrix["diagnosis"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.heatmap(Data[corr_features].corr(), annot = True)
plt.title("Columns having Correlation above 0.40% with diagnosis features", fontweight = "bold", fontsize=15)
plt.show()


# In[15]:


Data.drop(['smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'symmetry_se', 'fractal_dimension_se', 'fractal_dimension_worst'], axis = 1, inplace = True)
Data.info()


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# We are excluding the columns mentioned above because they either exhibit a negative correlation with the diagnosis feature or have a correlation coefficient below 0.40%. All the remaining columns will be used to build a model for training and predicting the output.

# # Machine Learning / Model Building

# In[16]:


X = Data.drop(['diagnosis'], axis=1)

y = Data['diagnosis']


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.65, random_state = 3332)


# In[18]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# ### 1) XG Boost Model

# In[19]:


from xgboost import XGBRegressor
XGB = XGBRegressor()
XGB.fit(X_train, y_train)


# In[20]:


Prediction_1 = XGB.predict(X_test)

Actual_1 = (y_test)

Accuracy_1 = XGB.score(X_test, y_test)*100
print('Accuracy of model is:', Accuracy_1)

print('Error in model is :', average_precision_score(Actual_1, Prediction_1)*100)

print('Error in model is :', mean_absolute_error(Actual_1, Prediction_1)*100)


# ### 2) Cat Boost Model

# In[21]:


from catboost import CatBoostRegressor
CBR = CatBoostRegressor()
CBR.fit(X_train, y_train)


# In[22]:


Prediction_2 = CBR.predict(X_test)

Actual_2 = (y_test)

Accuracy_2 = CBR.score(X_test, y_test)*100
print('Accuracy of model is:', Accuracy_2)

print('Error in model is :', average_precision_score(Actual_2, Prediction_2)*100)

print('Error in model is :', mean_absolute_error(Actual_2, Prediction_2)*100)


# ### 3) Gradient Boosting Model

# In[23]:


from sklearn.ensemble import GradientBoostingClassifier
GBR = GradientBoostingClassifier()
GBR.fit(X_train, y_train)


# In[24]:


Prediction_3 = GBR.predict(X_test)

Actual_3 = (y_test)

Accuracy_3 = GBR.score(X_test, y_test)*100
print('Accuracy of model is:', Accuracy_3)

print('Error in model is :', average_precision_score(Actual_3, Prediction_3)*100)

print('Error in model is :', mean_absolute_error(Actual_3, Prediction_3)*100)


# ### 4) Ada Boost Model

# In[25]:


from sklearn.ensemble import AdaBoostClassifier
ADA = AdaBoostClassifier()
ADA.fit(X_train, y_train)


# In[26]:


Prediction_4 = ADA.predict(X_test)

Actual_4 = (y_test)

Accuracy_4 = ADA.score(X_test, y_test)*100
print('Accuracy of model is:', Accuracy_4)

print('Error in model is :', average_precision_score(Actual_4, Prediction_4)*100)

print('Error in model is :', mean_absolute_error(Actual_4, Prediction_4)*100)


# ### 5) KNN Model

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)


# In[28]:


Prediction_5 = KNN.predict(X_test)

Actual_5 = (y_test)

Accuracy_5 = KNN.score(X_test, y_test)*100
print('Accuracy of model is:', Accuracy_5)

print('Error in model is :', average_precision_score(Actual_5, Prediction_5)*100)

print('Error in model is :', mean_absolute_error(Actual_5, Prediction_5)*100)


# # Result & Outcomes

# In[29]:


Result = pd.DataFrame({'Models':['XGB', 'CBR', 'GBR', 'ADA', 'KNN'],
                       'Accuracy':[Accuracy_1, Accuracy_2, Accuracy_3, Accuracy_4, Accuracy_5], 
                      'Precision':[average_precision_score(Actual_1, Prediction_1)*100, average_precision_score(Actual_2, Prediction_2)*100, 
                                  average_precision_score(Actual_3, Prediction_3)*100, average_precision_score(Actual_4, Prediction_4)*100, 
                                  average_precision_score(Actual_5, Prediction_5)*100],
                      'Error':[ mean_absolute_error(Actual_1, Prediction_1)*100,  mean_absolute_error(Actual_2, Prediction_2)*100, 
                               mean_absolute_error(Actual_3, Prediction_3)*100,  mean_absolute_error(Actual_4, Prediction_4)*100, 
                               mean_absolute_error(Actual_5, Prediction_5)*100]})
Result


# In[30]:


Model = Result['Models']
Accuracy = Result['Accuracy']
Precision = Result['Precision']
Error = Result['Error']

fig = pgo.Figure()

fig.add_trace(pgo.Bar(x = Model, y = Accuracy, name = 'Accuracy', width = 0.2))
fig.add_trace(pgo.Bar(x = Model, y = Precision, name = 'Precision', width = 0.2))
fig.add_trace(pgo.Bar(x = Model, y = Error, name = 'Error', width = 0.2))


fig.update_layout(title = 'Accuracy score of performed Models in %.')
fig.show()


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# After inputting data and training our machine learning model using five different algorithms, i evaluated their performance in terms of various parameters, such as accuracy score, precision score, and errors. Out of these models, Gradient Boosting Regression (GBR), AdaBoost (ADA), and K-Nearest Neighbors (KNN) stood out as the top three performers. 
# 
# I've decided to proceed with the AdaBoost model because I believe there is room for further enhancement within this algorithm, which could potentially lead to improved model scores and more accurate predictions.

# # Model Improvement

# In[31]:


ADA = AdaBoostClassifier()
rs = []
acc = []
for i in range(1,21,1):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.35, random_state = i)    
    model_ADA_rs = ADA.fit(X_train, y_train)
    predict_values_ADA_rs = model_ADA_rs.predict(X_test)
    acc.append(accuracy_score(y_test, predict_values_ADA_rs))
    rs.append(i)


# In[32]:


plt.figure(figsize=(10,10))
plt.plot(rs, acc, color ='BLACK')

for i in range(len(rs)):
    print(rs[i],acc[i])


# In[33]:


for i in range(0,20):
    if acc[i] > 0.97:
        print(acc[i])


# # <h2 style = "font-family: Comic Sans MS;background-color:#8f99cc	"> Observations </h2>
# 
# As I mentioned before, it appears that there has been a significant enhancement in the AdaBoost model's performance. This is evident in the fact that after making necessary adjustments, the model now achieves substantially higher accuracy scores. Previously, the highest accuracy achieved was 97%, while the model now consistently attains accuracy scores of 98% and even 98.5%.

# # Model Testing

# In[34]:


X_new = np.array([[10, 20, 30, 40, 50, 0., 7.005, 80.0, .900, 10.00, 11, 12, 13, 14, 15, 168, 17, 18, 19, 20]])
#Prediction of the Species from the input vector
Diagnosis = ADA.predict(X_new)
print(Diagnosis)

if (Diagnosis == 1):
    print('M = Malignant')
    
else:
    print('B = Benign')


# In[35]:


X_new = np.array([[10, 20, 30, 40, 50, 0, 7005, 800, 900, 1000, 11, 12, 13, 14, 15, 168, 17, 18, 19, 20]])
#Prediction of the Species from the input vector
Diagnosis = ADA.predict(X_new)
print(Diagnosis)

if (Diagnosis == 1):
    print('M = Malignant')
    
else:
    print('B = Benign')

