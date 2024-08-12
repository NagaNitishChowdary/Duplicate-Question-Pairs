# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:30:45 2024

@author: Naga Nitish
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('train.csv')
df
"""
            id  ...  is_duplicate
0            0  ...             0
1            1  ...             0
2            2  ...             0
3            3  ...             0
4            4  ...             0
       ...  ...           ...
404285  404285  ...             0
404286  404286  ...             1
404287  404287  ...             0
404288  404288  ...             0
404289  404289  ...             0

[404290 rows x 6 columns]
"""


new_df = df.sample(30000,random_state=2)
new_df
"""
            id  ...  is_duplicate
398782  398782  ...             1
115086  115086  ...             0
327711  327711  ...             0
367788  367788  ...             0
151235  151235  ...             0
       ...  ...           ...
243932  243932  ...             1
91980    91980  ...             0
266955  266955  ...             0
71112    71112  ...             1
312470  312470  ...             1

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

# =============================================================================


# Distribution of duplicate and non-duplicate questions 

new_df['is_duplicate'].value_counts()
"""
is_duplicate
0    19013
1    10987
"""


# =============================================================================


# Repeated Questions 
qid = pd.Series(new_df['qid1'].tolist() + new_df['qid2'].tolist())
print("Number of Unique questions: ", np.unique(qid).shape[0])
# 55299 ---> out of 60000 questions 

x = qid.value_counts() > 1
print("Number of questions getting repeated: ",x[x].shape[0])
# 3480 ---> out of 60000 questions


plt.hist(qid.value_counts().values,bins=160)
plt.yscale('log')
plt.show()

# =============================================================================


# Feature Engineering ---> Adding 7 features 

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


# Feature 1,2 ---> q1 length, q2 length

new_df['q1_len'] = new_df['question1'].str.len() 
new_df['q2_len'] = new_df['question2'].str.len()

new_df.head()
"""
            id    qid1    qid2  ... is_duplicate q1_len  q2_len
398782  398782  496695  532029  ...            1     76      77
115086  115086  187729  187730  ...            0     49      57
327711  327711  454161  454162  ...            0    105     120
367788  367788  498109  491396  ...            0     59     146
151235  151235  237843   50930  ...            0     35      50
"""

# =============================================================================

# Feature 3,4  ---> Number of words in question1, question2

new_df['q1_num_words'] = new_df['question1'].apply(lambda row:len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row:len(row.split(" ")))

new_df.head()
"""
            id    qid1    qid2  ... q2_len q1_num_words  q2_num_words
398782  398782  496695  532029  ...     77           12            12
115086  115086  187729  187730  ...     57           12            15
327711  327711  454161  454162  ...    120           25            17
367788  367788  498109  491396  ...    146           12            30
151235  151235  237843   50930  ...     50            5             9
"""


# =============================================================================

# Feature 5 

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))
    return len(w1 & w2)


new_df['words_common'] = new_df.apply(common_words, axis=1)

new_df.head()
"""
            id    qid1    qid2  ... q1_num_words q2_num_words  words_common
398782  398782  496695  532029  ...           12           12            11
115086  115086  187729  187730  ...           12           15             7
327711  327711  454161  454162  ...           25           17             2
367788  367788  498109  491396  ...           12           30             0
151235  151235  237843   50930  ...            5            9             3
"""

# =============================================================================

# Feature 6 

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))
    return len(w1) + len(w2)

new_df['word_total'] = new_df.apply(total_words,axis=1)

new_df.head()
"""
            id    qid1    qid2  ... q2_num_words words_common  word_total
398782  398782  496695  532029  ...           12           11          24
115086  115086  187729  187730  ...           15            7          23
327711  327711  454161  454162  ...           17            2          34
367788  367788  498109  491396  ...           30            0          32
151235  151235  237843   50930  ...            9            3          13
"""


# =============================================================================

# Feature 7 

new_df['word_share'] = round(new_df['words_common']/new_df['word_total'],2)
new_df.head()
"""
            id    qid1    qid2  ... words_common word_total  word_share
398782  398782  496695  532029  ...           11         24        0.46
115086  115086  187729  187730  ...            7         23        0.30
327711  327711  454161  454162  ...            2         34        0.06
367788  367788  498109  491396  ...            0         32        0.00
151235  151235  237843   50930  ...            3         13        0.23
"""

# =============================================================================

# ANALYSIS OF FEATURES 

sns.displot(new_df['q1_len'])
print("Minimum Characters: ", new_df['q1_len'].min())  # 2 
print("Maximum Characters: ", new_df['q1_len'].max())  # 391
print("Average Number of Characters: ", int(new_df['q1_len'].mean()))  # 59

sns.displot(new_df['q2_len'])
print("Minimum Characters: ", new_df['q2_len'].min())  # 6
print("Maximum Characters: ", new_df['q2_len'].max())  # 1151
print("Average Number of Characters: ", int(new_df['q2_len'].mean()))  # 60

# =============================================================================

sns.displot(new_df['q1_num_words'])
print("Minimum Words: ", new_df['q1_num_words'].min())  # 1
print("Maximum Words: ", new_df['q1_num_words'].max()) # 72
print("Average Number of Words: ",int(new_df['q1_num_words'].mean())) # 10

sns.displot(new_df['q2_num_words'])
print("Minimum Words: ", new_df['q2_num_words'].min())  # 1
print("Maximum Words: ", new_df['q2_num_words'].max()) # 237
print("Average Number of Words: ",int(new_df['q2_num_words'].mean())) # 11

# =============================================================================


ques_df = new_df[['question1','question2']]
ques_df
# [30000 rows x 2 columns]

final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
final_df
"""
(30000, 8)

        is_duplicate  q1_len  q2_len  ...  words_common  word_total  word_share
398782             1      76      77  ...            11          24        0.46
115086             0      49      57  ...             7          23        0.30
327711             0     105     120  ...             2          34        0.06
367788             0      59     146  ...             0          32        0.00
151235             0      35      50  ...             3          13        0.23
"""

# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer

# Merge Texts 
questions = list(ques_df['question1'].tolist() + list(ques_df['question2']))

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)

q1_arr
"""
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)

(30000, 3000)
"""


# =============================================================================

temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)

temp_df
"""
        0     1     2     3     4     5     ...  2994  2995  2996  2997  2998  2999
398782     0     0     0     0     0     0  ...     0     0     0     0     0     0
115086     0     0     0     0     0     0  ...     0     0     0     0     0     0
327711     0     0     0     0     0     0  ...     0     0     0     0     0     0
367788     0     0     0     0     0     0  ...     0     0     0     0     0     0
151235     0     0     0     0     0     0  ...     0     0     0     0     0     0

(30000, 6000)
"""

# =============================================================================

final_df = pd.concat([final_df,temp_df], axis=1)
final_df
"""
        is_duplicate  q1_len  q2_len  q1_num_words  ...  2996  2997  2998  2999
398782             1      76      77            12  ...     0     0     0     0
115086             0      49      57            12  ...     0     0     0     0
327711             0     105     120            25  ...     0     0     0     0
367788             0      59     146            12  ...     0     0     0     0
151235             0      35      50             5  ...     0     0     0     0
             ...     ...     ...           ...  ...   ...   ...   ...   ...
243932             1      42      41             7  ...     0     0     0     0
91980              0      68      61            12  ...     0     0     0     0
266955             0      73      98            14  ...     0     0     0     0
71112              1      51      45            10  ...     0     0     0     0
312470             1      87      77            15  ...     0     0     0     0

[30000 rows x 6008 columns]
"""

# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(final_df.iloc[:, 1:],final_df.iloc[:,0],test_size=0.2,random_state=42)


"""
Feature names are only supported if all input features have string names, 
but your input has ['int', 'str'] as feature name / column name types. 
If you want feature names to be stored and validated, you must convert 
them all to strings, by using X.columns = X.columns.astype(str)
"""
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train
"""
        q1_len  q2_len  q1_num_words  q2_num_words  ...  2996  2997  2998  2999
16886       87      73            16            13  ...     0     0     0     0
349845      53      46             8             8  ...     0     0     0     0
331239      52      45            10            10  ...     0     0     0     0
350540      35      28             6             4  ...     0     0     0     0
139497      22      31             4             5  ...     0     0     0     0
       ...     ...           ...           ...  ...   ...   ...   ...   ...
193120      62      61            12            12  ...     0     0     0     0
308620      35      49             7             8  ...     0     0     0     0
105504      38      22             7             5  ...     0     0     0     0
103393      76     146            17            30  ...     0     0     0     0
230141      22      24             3             4  ...     0     0     0     0

[24000 rows x 6007 columns]
"""

# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_Pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy_Score using Random Forest Classifier: ",accuracy_score(Y_Pred,Y_test).round(2))
# 0.77

# =============================================================================


# Accuracy improved by 3% by introducing sample features.
