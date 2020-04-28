#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pyprind
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer


# In[2]:


#开宇给大家的CODE开头我省略了，一些分词之类的，直接从他生成好的文件提取

count = CountVectorizer()
with open('movie_data.csv', encoding='utf-8', mode = 'r') as f:
    bag = count.fit_transform(f)
    print(count.vocabulary_)
    porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[3]:


nltk.download('stopwords')
from nltk.corpus import stopwords
#Classify the topics of movie set
df = pd.read_csv('movie_data.csv',encoding = 'utf-8')
count = CountVectorizer(stop_words = 'english',max_df=.15,max_features=5000)
X = count.fit_transform(df['review'].values)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 10, random_state = 0, learning_method='batch')
X_topics = lda.fit_transform(X)


# In[4]:


n_top_words = 30
feature_names = count.get_feature_names()
feature_vec = [None]
for topic_idx,topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx+1))
    print(" ".join([feature_names[i] for i in topic.argsort()        [:-n_top_words-1:-1]]))   
    feature_vec.append ((" ".join([feature_names[i] for i in topic.argsort()        [:-n_top_words-1:-1]]))) 

feature_vec.pop(0)
feature_vec = np.array(feature_vec)
feature_vec = count.fit_transform(feature_vec)
print(count.vocabulary_)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf = True,norm ='l2',smooth_idf = True)
np.set_printoptions(precision = 4)
feature_vec = tfidf.fit_transform(feature_vec)# coded feature vectors
print(feature_vec.toarray())


# In[5]:


print(X_topics)


# In[22]:


feature_vec[0][0]


# In[103]:


feature_list=feature_vec.toarray()
df_reclist=pd.read_excel('Rec_list.xlsx')

COS_list=[]
from scipy.spatial import distance

#这里的input_data_trained是随便生成的，实际中应该是用户INPUT之后模型训练完的一个特征向量
#注意需要大小为 1 X 135
input_data_trained=np.zeros(135) 
input_data_trained[5:20]=0.995

i=0
b=input_data_trained.reshape(-1,1)

# 为了保证余弦对数据敏感，用a和b减去自身的平均数后再算余弦相似度（1-COSINE DISTANCE)
b=b-b.mean()
while i<10:
    a=feature_list[i].reshape(-1, 1)
    a=a-a.mean()
    COS_list.append(1-distance.cosine(a,b))
    i=i+1
    
index_num=COS_list.index(max(COS_list))
print("Recommendations: ")
print(df_reclist.iloc[:,index_num])



