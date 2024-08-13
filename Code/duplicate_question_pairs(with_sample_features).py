# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:27:38 2024

@author: Naga Nitish
"""

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


# =============================================================================

"""

TOKENS, WORDS , STOPWORDS 

SENTENCE = "Nitish is a good boy"
Tokens = [Nitish] , [is] , [a] , [good] , [boy]
Words = Tokens - StopWords = [Nitish] , [good] , [boy]
StopWords = [is] , [a]

"""


"""
        ADVANCED FEATURES
        ----------------

1) CWC_MIN ---> COMMON WORDS
                ------------
              MIN(WORDS(Q1,Q2))

2) CWC_MAX ---> COMMON WORDS 
                ------------
              MAX(WORDS(Q1,Q2))

3) CSC_MIN ---> COMMON STOPWORDS
                ----------------
               MIN(STOPWORDS(Q1,Q2))
               
4) CSC_MAX ---> COMMON STOPWORDS
                ----------------
               MAX(STOPWORDS(Q1,Q2))
               
5) CTC_MIN ---> COMMON TOKENS
                -------------
               MIN(TOKENS(Q1,Q2))
               
6) CTC_MAX ---> COMMON TOKENS
                -------------
               MAX(TOKENS(Q1,Q2))

7) LAST_WORD_EQUAL 

8) FIRST_WORD_EQUAL

"""


"""

    LENGTH BASED FEATURES
    ---------------------
    
1) MEAN_LENGTH ---> MEAN OF THE LENGTH OF 2 QUESTIONS(TOKENS)

2) ABS_LEN_DIFF 

3) LONGEST_SUBSTR_RATIO    

"""


"""
    
    FUZZY FEATURES
    --------------

1) FUZZ_RATIO ---> FUZZ_RATIO SCORE FROM FUZZYWUZZY(LIBRARY)

2) FUZZ_PARTIAL_RATIO ---> FUZZ_PARTIAL_RATIO FROM FUZZYWUZZY

3) TOKEN_SORT_RATIO ---> TOKEN_SORT_RATIO FROM FUZZYWUZZY

4) TOKEN_SET_RATIO ---> TOKEN_SET_RATIO FROM FUZZYWUZZY 

"""

# =============================================================================


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

# =============================================================================

def preprocess(q):
    q = str(q).lower().strip() 
    
    # REPLACE CERTAIN SPECIAL CHARACTERS WITH THEIR STRING EQUIVALENTS 
    q = q.replace('%', 'percent')
    q = q.replace('$' , 'dollar')
    q = q.replace('@' , 'at')
    
    # THE PATTERN '[math]' APPEARS AROUND 900 TIMES IN THE WHOLE DATASET
    q = q.replace('[math]','')
    
    # REPLACING SOME NUMBERS WITH STRING EQUIVALENTS (NOT PERFECT, CAN BE DONE BETTER TO ACCOUNT FOR MORE CASES)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace('000 ', 'k ')
    
    # DECONTRACTING WORDS 
    # 
    contractions = {
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
        }
    
    q_decontracted = [] 
    
    for word in q.split():
        if word in contractions:
            word = contractions[word]
            
        q_decontracted.append(word)
        
    
    q = ' '.join(q_decontracted)
    q = q.replace("'ve" , " have")
    return q     


# Testing the preprocess function 
preprocess("I've already! wasn't ?")
# 'i have already! was not ?'


# =============================================================================

new_df.columns
"""
['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
"""


new_df['question1'] = new_df['question1'].apply(preprocess)
new_df['question2'] = new_df['question2'].apply(preprocess)

# =============================================================================


new_df['q1_len'] = new_df['question1'].str.len()
new_df['q2_len'] = new_df['question2'].str.len()

# =============================================================================


new_df['q1_num_words'] = new_df['question1'].apply(lambda row:len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row:len(row.split(" ")))

# =============================================================================

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))
    return len(w1 & w2)


new_df['words_common'] = new_df.apply(common_words, axis=1)

# =============================================================================


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

# ADVANCED FEATURES 

from nltk.corpus import stopwords

def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001
    
    STOP_WORDS = stopwords.words('english')
    
    # IF DATA DOESN'T CAME PROPERLY OR IS THERE ANY PROBLEM JUST RETURN TOKEN_FREATURES
    token_features = [0.0]*8    
    
    # CONVERTING THE SENTENCE INTO TOKENS 
    
    q1_tokens = q1.split()
    q2_tokens = q2.split() 
    
    if(len(q1_tokens) == 0 or len(q2_tokens) == 0):
        return token_features
    
    # GET THE NON-STOPWORDS FROM QUESTIONS 
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    # GET THE STOPWORDS IN QUESTIONS 
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    
    # GET THE COMMON NON-STOPWORDS FROM QUESTION PAIR
    common_word_count = len(q1_words.intersection(q2_words))
    
    # GET THE COMMON STOPWORDS FROM QUESTION PAIR 
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # GET THE COMMON TOKENS FROM QUESTION PAIRS
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count/(min(len(q1_words) , len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count/(max(len(q1_words) , len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count/(min(len(q1_stops) , len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count/(max(len(q1_stops) , len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count/(min(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count/(max(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
    
    # LAST WORD OF BOTH QUESTION IS SAME OR NOT
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # FIRST WORD OF SAME QUESTION IS SAME OR NOT 
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features


token_features = new_df.apply(fetch_token_features,axis=1)
token_features
"""
398782    [0.8571306124198226, 0.8571306124198226, 0.999...
115086    [0.7499812504687383, 0.5999880002399952, 0.666...
327711    [0.0, 0.0, 0.3999920001599968, 0.2222197531138...
367788                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]
151235    [0.7499812504687383, 0.5999880002399952, 0.0, ...
                       
243932    [0.7499812504687383, 0.7499812504687383, 0.999...
91980     [0.4285653062099113, 0.4285653062099113, 0.199...
266955    [0.249996875039062, 0.2222197531138543, 0.3333...
71112     [0.49998750031249223, 0.3999920001599968, 0.59...
312470    [0.39999600003999963, 0.39999600003999963, 0.2...
Length: 30000, dtype: object
"""

new_df['cwc_min'] = list(map(lambda x:x[0] , token_features))
new_df['cwc_max'] = list(map(lambda x:x[1] , token_features))
new_df['csc_min'] = list(map(lambda x:x[2] , token_features))
new_df['csc_max'] = list(map(lambda x:x[3] , token_features))
new_df['ctc_min'] = list(map(lambda x:x[4] , token_features))
new_df['ctc_max'] = list(map(lambda x:x[5] , token_features))
new_df['last_word_eq'] = list(map(lambda x:x[6] , token_features))
new_df['first_word_eq'] = list(map(lambda x:x[7] , token_features))

new_df
"""
            id    qid1    qid2  ...   ctc_max last_word_eq  first_word_eq
398782  398782  496695  532029  ...  0.916659            1              1
115086  115086  187729  187730  ...  0.466664            1              1
327711  327711  454161  454162  ...  0.080000            0              0
367788  367788  498109  491396  ...  0.000000            0              0
151235  151235  237843   50930  ...  0.333330            1              0
       ...     ...     ...  ...       ...          ...            ...
243932  243932   26193  356455  ...  0.857131            0              1
91980    91980  154063  154064  ...  0.249998            0              0
266955  266955  133017  384210  ...  0.235293            0              0
71112    71112  122427  122428  ...  0.499995            0              1
312470  312470  436915  436916  ...  0.333331            0              0
"""

# =============================================================================


# Length Features
# ---------------

# pip install distance
import distance

def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if(len(q1_tokens) == 0 or len(q2_tokens) == 0):
        return length_features
    
    
    # ABSOLUTE LENGTH FEATURES 
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # AVERAGE TOKEN LENGTH OF BOTH QUESTIONS 
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2 
    
    strs = list(distance.lcsubstrings(q1,q2))
    
    if(len(strs) != 0):
        length_features[2] = len(strs[0]) / (min(len(q1) , len(q2)) + 1)
    
    return length_features

length_features = new_df.apply(fetch_length_features,axis=1)

new_df['abs_len_diff'] = list(map(lambda x : x[0], length_features))
new_df['mean_len'] = list(map(lambda x : x[1], length_features))
new_df['longest_substr_ratio'] = list(map(lambda x : x[2], length_features))


new_df
"""
            id    qid1    qid2  ... abs_len_diff mean_len  longest_substr_ratio
398782  398782  496695  532029  ...            0     12.0              0.844156
115086  115086  187729  187730  ...            3     13.5              0.220000
327711  327711  454161  454162  ...            8     21.0              0.047170
367788  367788  498109  491396  ...           18     21.0              0.050000
151235  151235  237843   50930  ...            4      7.0              0.555556
       ...     ...     ...  ...          ...      ...                   ...
243932  243932   26193  356455  ...            0      7.0              0.761905
91980    91980  154063  154064  ...            4     14.0              0.260870
266955  266955  133017  384210  ...            3     15.5              0.135135
71112    71112  122427  122428  ...            0     10.0              0.282609
312470  312470  436915  436916  ...            1     14.5              0.192308

[30000 rows x 24 columns]
"""

# =============================================================================


# FUZZY FEATURES
# --------------

# pip install fuzzywuzzy
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # FUZZY RATIO
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    # FUZZY PARTIAL RATIO 
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)
    
    # TOKEN_SORT_RATIO 
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    # TOKEN_SET_RATIO
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)
    
    return fuzzy_features 


fuzzy_features = new_df.apply(fetch_fuzzy_features,axis=1)


# Creating new feature columns for fuzzy features
new_df['fuzz_ratio'] = list(map(lambda x:x[0] , fuzzy_features))
new_df['fuzz_partial_ratio'] = list(map(lambda x:x[1] , fuzzy_features))
new_df['token_sort_ratio'] = list(map(lambda x:x[2] , fuzzy_features))
new_df['token_set_ratio'] = list(map(lambda x:x[3] , fuzzy_features))

new_df
"""
            id    qid1  ...  token_sort_ratio token_set_ratio
398782  398782  496695  ...                99              99
115086  115086  187729  ...                65              74
327711  327711  454161  ...                34              43
367788  367788  498109  ...                23              30
151235  151235  237843  ...                48              69
       ...     ...  ...               ...             ...
243932  243932   26193  ...                77              89
91980    91980  154063  ...                49              51
266955  266955  133017  ...                56              67
71112    71112  122427  ...                72              74
312470  312470  436915  ...                61              62

[30000 rows x 28 columns]
"""

# =============================================================================


# USING TNSE FOR DIMENSIONALITY REDUCTION FOR 15 FEATURES
# (GENERATED AFTER CLEANING THE DATA) TO 3 DIMENSION


from sklearn.preprocessing import MinMaxScaler 

X = MinMaxScaler().fit_transform(new_df[['cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max','last_word_eq','first_word_eq','abs_len_diff','mean_len','longest_substr_ratio','fuzz_ratio','fuzz_partial_ratio','token_sort_ratio','token_set_ratio']])
y = new_df['is_duplicate'].values


from sklearn.manifold import TSNE

tsne2d = TSNE(
    n_components=2,
    init='random',  #PCA
    random_state = 101,
    method = 'barnes_hut',
    n_iter = 1000,
    verbose = 2,
    angle=0.5
    ).fit_transform(X)


tsne3d = TSNE(
    n_components = 3,
    init='random',  #PCA
    random_state = 101,
    method = 'barnes_hut',
    n_iter = 1000,
    verbose = 2,
    angle=0.5
    ).fit_transform(X)


# =============================================================================

final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
final_df.head()
"""
        is_duplicate  q1_len  ...  token_sort_ratio  token_set_ratio
398782             1      76  ...                99               99
115086             0      49  ...                65               74
327711             0     105  ...                34               43
367788             0      59  ...                23               30
151235             0      35  ...                48               69
""" 

ques_df = new_df[['question1','question2']]
ques_df.head()
"""
                                                question1                                          question2
398782  what is the best marketing automation tool for...  what is the best marketing automation tool for...
115086  i am poor but i want to invest. what should i do?  i am quite poor and i want to be very rich. wh...
327711  i am from india and live abroad. i met a guy f...  t.i.e.t to thapar university to thapar univers...
367788  why do so many people in the u.s. hate the sou...  my boyfriend doesnt feel guilty when he hurts ...
151235                consequences of bhopal gas tragedy?  what was the reason behind the bhopal gas trag...
"""

# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer

# Merge Texts 
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)

q1_arr.shape
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
     ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...   ...
243932     0     0     0     0     0     0  ...     0     0     0     0     0     0
91980      0     0     0     0     0     0  ...     0     0     0     0     0     0
266955     0     0     0     0     0     0  ...     0     0     0     0     0     0
71112      0     0     0     0     0     0  ...     0     0     0     0     0     0
312470     0     0     0     0     1     0  ...     0     0     0     0     0     0

[30000 rows x 6000 columns]
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
91980              0      68      71            12  ...     0     0     0     0
266955             0      73      98            14  ...     0     0     0     0
71112              1      51      45            10  ...     0     0     0     0
312470             1      87      77            15  ...     0     0     0     0

[30000 rows x 6023 columns]
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

# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_Pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy_Score using Random Forest Classifier: ",accuracy_score(Y_Pred,Y_test).round(2))
# 0.79  ---> more better after adding the Advanced Features

# =============================================================================

"""
          Predict
          0     1

       0  ok  not_ok
Actual
       1  ok   ok 


If the questions are NOT DUPLICATE, but if our model predicate as DUPLICATE
and if we merges, then user experience gets worst.

"""

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,Y_Pred)
"""
array([[3154,  645],
       [ 595, 1606]], dtype=int64)
"""

# Here 645 predictions are dangerous

# Finally we got roughly 80% ACCURACY 

# HERE WE HAVE TAKEN ONLY 30000 SAMPLES, WE CAN INCREASE MORE SAMPLES, 
# THEN ACCURACY WILL BE INCREASED. 


# =============================================================================


# we can increase accuracy by 

# 1) Increase data 
# 2) preprocessing ---> stemming
# 3) Apply more algorithms ---> SVM logistic, Hyper parameter tuning , cross validation
# 4) More Features 
# 5) Bag of Words ---> tfidf, word2vec, tfidf weighted w2v
# 6) Apply deep learning( LSTM neural network, transformers)


# =============================================================================

# Prediction Part wheather the 2 questions are same or not 

def test_common_words(w1,w2):
    w1 = set(map(lambda word: word.lower().strip(), w1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , w2.split(" ")))
    return len(w1 & w2)

def test_total_words(w1,w2):
    w1 = set(map(lambda word: word.lower().strip(), w1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , w2.split(" ")))
    return len(w1) + len(w2)

def test_fetch_token_features(q1,q2):
    SAFE_DIV = 0.0001
    
    STOP_WORDS = stopwords.words('english')
    
    # IF DATA DOESN'T CAME PROPERLY OR IS THERE ANY PROBLEM JUST RETURN TOKEN_FREATURES
    token_features = [0.0]*8    
    
    # CONVERTING THE SENTENCE INTO TOKENS 
    
    q1_tokens = q1.split()
    q2_tokens = q2.split() 
    
    if(len(q1_tokens) == 0 or len(q2_tokens) == 0):
        return token_features
    
    # GET THE NON-STOPWORDS FROM QUESTIONS 
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    # GET THE STOPWORDS IN QUESTIONS 
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    
    # GET THE COMMON NON-STOPWORDS FROM QUESTION PAIR
    common_word_count = len(q1_words.intersection(q2_words))
    
    # GET THE COMMON STOPWORDS FROM QUESTION PAIR 
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # GET THE COMMON TOKENS FROM QUESTION PAIRS
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count/(min(len(q1_words) , len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count/(max(len(q1_words) , len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count/(min(len(q1_stops) , len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count/(max(len(q1_stops) , len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count/(min(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count/(max(len(q1_tokens) , len(q2_tokens)) + SAFE_DIV)
    
    # LAST WORD OF BOTH QUESTION IS SAME OR NOT
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # FIRST WORD OF SAME QUESTION IS SAME OR NOT 
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

def test_fetch_length_features(q1,q2):
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if(len(q1_tokens) == 0 or len(q2_tokens) == 0):
        return length_features
    
    
    # ABSOLUTE LENGTH FEATURES 
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # AVERAGE TOKEN LENGTH OF BOTH QUESTIONS 
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2 
    
    strs = list(distance.lcsubstrings(q1,q2))
    
    if(len(strs) != 0):
        length_features[2] = len(strs[0]) / (min(len(q1) , len(q2)) + 1)
    
    return length_features


def test_fetch_fuzzy_features(q1,q2):
    
    fuzzy_features = [0.0]*4
    
    # FUZZY RATIO
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    # FUZZY PARTIAL RATIO 
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)
    
    # TOKEN_SORT_RATIO 
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    # TOKEN_SET_RATIO
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)
    
    return fuzzy_features 

def query_point_creator(q1,q2):
    input_query = []  # stores all 22 features
    
    # preprocess 
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # Fetch basic features 
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(test_common_words(q1,q2))
    input_query.append(test_total_words(q1,q2))
    input_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))
    
    
    # FETCH TOKEN FEATURES 
    token_features = test_fetch_token_features(q1,q2)
    input_query.extend(token_features)
    
    
    # FETCH LENGTH BASED FEATURES 
    length_features = test_fetch_length_features(q1,q2)
    input_query.extend(length_features)
    
    
    # FETCH FUZZY FEATURES 
    fuzzy_features = test_fetch_fuzzy_features(q1,q2)
    input_query.extend(fuzzy_features)
    
    # bow(bag of words) feature for q1
    q1_bow = cv.transform([q1]).toarray()
    
    # bow(bag of words) feature for q2 
    q2_bow = cv.transform([q2]).toarray() 
    
    
    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))


q1 = "Where is the capital of India?"
q2 = "What is the current capital of India?"


query_point_creator(q1,q2)
# array([[30., 37.,  6., ...,  0.,  0.,  0.]])
# (1, 6022)


rfc.predict(query_point_creator(q1,q2))
# array([1], dtype=int64)   ---> same questions 

# =============================================================================

q3 = "Where is the capital of India?"
q4 = "What is the current capital of Pakistan?"

rfc.predict(query_point_creator(q3,q4))
# array([0], dtype=int64) ---> different questions 

# =============================================================================
