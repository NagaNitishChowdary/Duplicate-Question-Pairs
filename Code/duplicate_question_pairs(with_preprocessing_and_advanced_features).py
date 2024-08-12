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


