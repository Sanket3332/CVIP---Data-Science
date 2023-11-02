#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import plotly.express as px
import plotly.graph_objects as pgo

import warnings;
warnings.filterwarnings('ignore')


# # Data loading & Manipulating

# In[2]:


email = pd.read_excel('D:\Disk F\My Stuff\Internships\OASIS INFOBYTE\TASK 4\Dataset.xlsx')
email


# In[3]:


email.columns


# In[4]:


email.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

email.rename(columns = {'v1':'Category', 'v2':'Email'}, inplace=True)
email.head()


# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

email['Category'] = le.fit_transform(email['Category'])
email.head()


# In[6]:


email = email.drop_duplicates(keep = 'first')


# In[7]:


email['Email'] = email['Email'].apply(str)


# In[8]:


email['Email'].dtypes


# In[9]:


email['Category'].dtypes


# In[10]:


email.info()


# In[11]:


email.describe()


# In[12]:


email.isna().sum()


# In[13]:


email.duplicated().sum()


# In[14]:


email = email.drop_duplicates(keep = 'first')


# In[15]:


email.duplicated().sum()


# In[16]:


email.Category.value_counts()


# # Data Visualization

# In[17]:


plt.pie(email['Category'].value_counts(), explode = (0.15, 0), labels = ('ham', 'spam'), colors = ('green', 'red'), autopct = '%2.2f%%', shadow = True, startangle = 19)
plt.title(" Percentages of Emails")
plt.show()


# In[18]:


from wordcloud import WordCloud

WC = WordCloud(width = 350, height = 350, min_font_size = 8, background_color = 'black')


# In[19]:


ham = WC.generate(email[email['Category'] == 0]['Email'].str.cat(sep = ' '))

plt.imshow(ham);


# In[56]:


spam = WC.generate(email[email['Category'] == 1]['Email'].str.cat(sep = ' '))

plt.imshow(spam);


# # Machine Laerning Algorithmn

# In[21]:


X = email['Email']

y = email['Category']


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.72, random_state = 3332)


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer

feature_extraction = CountVectorizer(min_df = 1, stop_words = 'english')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


# # 1) Multinomial Naive Bayes Model 

# In[24]:


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()


# In[25]:


MNB.fit(X_train_features, y_train)


# In[26]:


MNB_Accu = MNB.score(X_test_features, y_test) * 100
print('Accuracy:', MNB_Accu)

MNB_pred = MNB.predict(X_test_features)
print(MNB_pred)

Actual1 = y_test
print(Actual1)


# In[27]:


from sklearn.metrics import precision_score
print('Precision Score:', precision_score(Actual1, MNB_pred)*100)


# # 2) Bernoulli Naive Bayes Model

# In[28]:


from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()


# In[29]:


BNB.fit(X_train_features, y_train)


# In[30]:


BNB_Accu = BNB.score(X_test_features, y_test) * 100
print('Accuracy:', BNB_Accu)

BNB_pred = BNB.predict(X_test_features)
print(BNB_pred)

Actual2 = y_test
print(Actual2)


# In[31]:


from sklearn.metrics import precision_score
print('Precision Score:', precision_score(Actual2, BNB_pred)*100)


# # Results

# In[32]:


Result = pd.DataFrame({'Models':['MNB', 'BNB'],
             'Accuracy':[MNB_Accu, BNB_Accu],
             'Precision':[precision_score(Actual1, MNB_pred)*100, precision_score(Actual2, BNB_pred)*100]})

Result


# In[33]:


Model = Result['Models']

Model_Accuracy = Result['Accuracy']
Model_Precision = Result['Precision']

fig = pgo.Figure()
fig.add_trace(pgo.Bar(x = Model, y = Model_Accuracy, name = 'Accuracy', width = 0.3))
fig.add_trace(pgo.Bar(x = Model, y = Model_Precision, name = 'Precision', width = 0.2))

fig.update_layout(title = 'Accuracy & Precision score of performed Models in %.')

fig.show()


# # Model Testing (Email Classifier)

# In[54]:


email['Email'][258]


# In[55]:


Input_Email = ['We tried to contact you re your reply to our offer of a Video Handset? 750 anytime networks mins? UNLIMITED TEXT? Camcorder? Reply or call 08000930705 NOW']

input_data_features = feature_extraction.transform(Input_Email)

prediction = MNB.predict(input_data_features)

print(prediction)

if (prediction == 0):
    print('Ham mail')
    
else:
    print('Spam Mail')
    


# In[ ]:




