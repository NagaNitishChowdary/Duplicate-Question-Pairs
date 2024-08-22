# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:24:39 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv('train.csv')
df

#             id  ...  is_duplicate
# 0            0  ...             0
# 1            1  ...             0
# 2            2  ...             0
# 3            3  ...             0
# 4            4  ...             0
#        ...  ...           ...
# 404285  404285  ...             0
# 404286  404286  ...             1
# 404287  404287  ...             0
# 404288  404288  ...             0
# 404289  404289  ...             0

# [404290 rows x 6 columns]

df.head()
#    id  qid1  ...                                          question2 is_duplicate
# 0   0     1  ...  What is the step by step guide to invest in sh...            0
# 1   1     3  ...  What would happen if the Indian government sto...            0
# 2   2     5  ...  How can Internet speed be increased by hacking...            0
# 3   3     7  ...  Find the remainder when [math]23^{24}[/math] i...            0
# 4   4     9  ...            Which fish would survive in salt water?            0

df.columns
# ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']


df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 404290 entries, 0 to 404289
Data columns (total 6 columns):
 #   Column        Non-Null Count   Dtype 
---  ------        --------------   ----- 
 0   id            404290 non-null  int64 
 1   qid1          404290 non-null  int64 
 2   qid2          404290 non-null  int64 
 3   question1     404289 non-null  object
 4   question2     404288 non-null  object
 5   is_duplicate  404290 non-null  int64 
dtypes: int64(4), object(2) 
"""

df.describe()
"""
                  id           qid1           qid2   is_duplicate
count  404290.000000  404290.000000  404290.000000  404290.000000
mean   202144.500000  217243.942418  220955.655337       0.369198
std    116708.614502  157751.700002  159903.182629       0.482588
min         0.000000       1.000000       2.000000       0.000000
25%    101072.250000   74437.500000   74727.000000       0.000000
50%    202144.500000  192182.000000  197052.000000       0.000000
75%    303216.750000  346573.500000  354692.500000       1.000000
max    404289.000000  537932.000000  537933.000000       1.000000
"""

# =============================================================================

# Check if there are any null values 
df.isnull().sum()
"""
id              0
qid1            0
qid2            0
question1       1
question2       2
is_duplicate    0
"""

# =============================================================================


# Check if there any duplicate rows

df.duplicated().sum()
# 0

# =============================================================================

# Distribution of Non-duplicate and Duplicate questions 

print(df['is_duplicate'].value_counts()) 
"""
is_duplicate
0    255027  ---> 63.1 %
1    149263  ---> 36.9 %
"""

# Bar graph
df['is_duplicate'].value_counts().plot(kind='bar')


# =============================================================================


# Repeated Questions 
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print("Number of Unique questions: ", np.unique(qid).shape[0])
# 537933 ---> out of 8 lakh questions 

x = qid.value_counts() > 1
print("Number of questions getting repeated: ",x[x].shape[0])
# 111780 ---> out of 8 lakh questions


plt.hist(qid.value_counts().values,bins=160)
plt.yscale('log')
plt.show()

# =============================================================================


"""
**** AS THERE ARE 4 LAKH ROWS, IT IS VERY HARD TO RUN IN THE SYSTEM, 
LET'S TAKE THE RANDOM SAMPLE OF 30000 ROWS 
"""


new_df = df.sample(30000)
new_df
"""
            id  ...  is_duplicate
95650    95650  ...             0
392370  392370  ...             0
176416  176416  ...             0
307262  307262  ...             0
2349      2349  ...             0
       ...  ...           ...
339704  339704  ...             1
85179    85179  ...             0
2174      2174  ...             1
251163  251163  ...             1
328834  328834  ...             0

[30000 rows x 6 columns]
"""

new_df.isnull().sum()
"""
id              0
qid1            0
qid2            0
question1       0
question2       0
is_duplicate    0
dtype: int64
"""

new_df.duplicated().sum()
# 0

print(df['is_duplicate'].value_counts()) 
"""
is_duplicate
0    255027
1    149263
Name: count, dtype: int64
"""

# =============================================================================

ques_df = new_df[['question1','question2']]
ques_df

"""
                                                question1                                          question2
30005      Why should I upgrade my iPhone 5 to iPhone 5s?  Should I upgrade to the iPhone 6 if I have an ...
167354  I want to learn programming. Is MIT OpenCourse...  I feel someone has hacked into my phone. How c...
350640                Why aren't Apple’s USB3 ports blue?  Why aren't USB3 ports on Macs blue, like on mo...
198339  What's the meaning of the song 'my youth is yo...  What is your favorite Beatles song? And why do...
200177  If elected president, should Hillary Clinton p...  If Hillary Clinton wins, would she nominate Ba...
                                                  ...                                                ...
395276        Which is the best joke you have ever heard?  What are some of funniest jokes that you have ...
171614  Is it possible to survive with a gross salary ...  Is it possible to survive with a gross salary ...
56577                     What are the GD topics for IIT?                           What are some GD topics?
346838            Which state in Nepal speaks pure Hindi?                     Which state speaks pure Hindi?
323396                               Why are people lazy?                          Why are some people lazy?

[30000 rows x 2 columns]
"""

# =============================================================================


from sklearn.feature_extraction.text import CountVectorizer 
# merge texts 
questions = list(ques_df['question1']) + list(ques_df['question2'])
questions[0]
# 'Why should I upgrade my iPhone 5 to iPhone 5s?'
questions[1]
# 'I want to learn programming. Is MIT OpenCourseWare a good choice?'
len(questions)
# 60000

"""
The CountVectorizer() method in NLP is used to convert text data into a 
numerical representation that can be processed by machine learning algorithms. 
This process is known as vectorization.

Here's how it works:
Tokenization: The CountVectorizer() method first breaks down the text into 
------------
individual words or tokens.

Vocabulary Creation: It then creates a vocabulary of all unique words 
-------------------
encountered in the text data.

Vectorization: For each document in the dataset, a numerical vector is created. 
-------------
The length of the vector is equal to the size of the vocabulary. Each element 
in the vector represents the frequency of a corresponding word in the document.
"""
cv = CountVectorizer(max_features=3000) # e.g. considering 3000 words
"""
Why is it used?

Machine Learning Compatibility: Most machine learning algorithms require 
-------------------------------
numerical data as input. The CountVectorizer() method transforms text data 
into a numerical format that can be used by these algorithms.

Feature Engineering: It can be used as a feature engineering technique to 
------------------
extract meaningful features from text data. The frequency of words can provide 
valuable information about the content of a document.

Similarity Measures: The numerical representation can be used to calculate 
-------------------
similarity between documents based on their word frequencies. This is useful 
for tasks like document clustering and recommendation systems.
"""

q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)

# cv.fit_transform(questions).toarray().shape   
# (60000, 3000)
q1_arr.shape  # (30000, 3000)
q2_arr.shape  # (30000, 3000)

q1_arr
"""
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
"""

# Number of 1's in the first row
np.sum(q1_arr[0] == 1) # 10

# =============================================================================

temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index) 
temp_df1
"""
        0     1     2     3     4     5     ...  2994  2995  2996  2997  2998  2999
95650      0     0     0     0     0     0  ...     0     0     0     0     0     0
392370     0     0     0     0     0     0  ...     0     0     0     0     0     0
176416     0     0     0     0     0     1  ...     0     0     0     0     0     0
307262     0     0     0     0     0     0  ...     0     0     0     0     0     0
2349       0     0     0     0     0     0  ...     0     0     0     0     0     0
     ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...   ...
339704     0     0     0     0     0     0  ...     0     0     0     0     0     0
85179      0     0     0     0     0     0  ...     0     0     0     0     0     0
2174       0     0     0     0     0     0  ...     0     0     0     0     0     0
251163     0     0     0     0     0     0  ...     0     0     0     0     0     0
328834     0     0     0     0     0     0  ...     0     0     0     0     0     0

[30000 rows x 3000 columns]   ---> 3000 words
"""

temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df2
"""
        0     1     2     3     4     5     ...  2994  2995  2996  2997  2998  2999
95650      0     0     0     0     0     0  ...     0     0     0     0     0     0
392370     0     0     0     0     0     0  ...     0     0     0     0     0     0
176416     0     0     0     0     0     1  ...     0     0     0     0     0     0
307262     0     0     0     0     0     0  ...     0     0     0     0     0     0
2349       0     0     0     0     0     0  ...     0     0     0     0     0     0
     ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...   ...
339704     0     0     0     0     0     0  ...     0     0     0     0     0     0
85179      0     0     0     0     0     0  ...     0     0     0     0     0     0
2174       0     0     0     0     0     0  ...     0     0     0     0     0     0
251163     0     0     0     0     0     0  ...     0     0     0     0     0     0
328834     0     0     0     0     0     0  ...     0     0     0     0     0     0

[30000 rows x 3000 columns]
"""

temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df
"""
        0     1     2     3     4     5     ...  2994  2995  2996  2997  2998  2999
95650      0     0     0     0     0     0  ...     0     0     0     0     0     0
392370     0     0     0     0     0     0  ...     0     0     0     0     0     0
176416     0     0     0     0     0     1  ...     0     0     0     0     0     0
307262     0     0     0     0     0     0  ...     0     0     0     0     0     0
2349       0     0     0     0     0     0  ...     0     0     0     0     0     0
     ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...   ...
339704     0     0     0     0     0     0  ...     0     0     0     0     0     0
85179      0     0     0     0     0     0  ...     0     0     0     0     0     0
2174       0     0     0     0     0     0  ...     0     0     0     0     0     0
251163     0     0     0     0     0     0  ...     0     0     0     0     0     0
328834     0     0     0     0     0     0  ...     0     0     0     0     0     0

[30000 rows x 6000 columns]   ---> 6000 words
"""

temp_df['is_duplicate'] = new_df['is_duplicate']
temp_df
"""
        0  1  2  3  4  5  6  ...  2994  2995  2996  2997  2998  2999  is_duplicate
95650   0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0
392370  0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0
176416  0  0  0  0  0  1  0  ...     0     0     0     0     0     0             0
307262  0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0
2349    0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0
   .. .. .. .. .. .. ..  ...   ...   ...   ...   ...   ...   ...           ...
339704  0  0  0  0  0  0  0  ...     0     0     0     0     0     0             1
85179   0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0
2174    0  0  0  0  0  0  0  ...     0     0     0     0     0     0             1
251163  0  0  0  0  0  0  0  ...     0     0     0     0     0     0             1
328834  0  0  0  0  0  0  0  ...     0     0     0     0     0     0             0

[30000 rows x 6001 columns]
"""

# =============================================================================


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(temp_df.iloc[:,0:-1].values, temp_df.iloc[:,-1].values, test_size=0.2, random_state=42)

X_train
"""
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 1, 0, ..., 0, 0, 0]], dtype=int64)

[24000, 6000]
"""

# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_Pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy_Score using Random Forest Classifier: ",accuracy_score(Y_Pred,Y_test).round(2))
# 0.74

# =============================================================================

# !pip install xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)

Y_Pred = xgb.predict(X_test)
print("Accuracy_score using XGB Classifier: ",accuracy_score(Y_Pred, Y_test).round(2))
# 0.72

# =============================================================================

# Using the common methods we obtained accuracy scores as 74% and 73%
# Some here we introduced some features 


# qid1 | qid2 | question1 | question2 | is_duplicate

# FEATURES
# --------
# 1 ---> q1 length 
# 2 ---> q2 length
# 3 ---> q1 words (Number of words in question1)
# 4 ---> q2 words (Number of words in question2)
# 5 ---> words common 
# 6 ---> words total (Total words in question1 + Total words in question2)
# 7 ---> words share(word common/word total)

# For every row, here we are introducing 7 new columns 

# =============================================================================
