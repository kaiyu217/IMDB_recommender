import numpy as np
import pyprind
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

basepath = 'E:\IMDB'
labels = {'pos':1,'neg':0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in('test','train'):
    for l in('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),
                'r',encoding='utf-8')as infile:
                txt = infile.read()
                df = df.append([[txt,labels[l]]],ignore_index =True )
            pbar.update()

df.columns = ['review','sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index = False,encoding = 'utf-8')
count = CountVectorizer()
doc = open('E:/Machine Learning/final/movie_data.csv','r')
bag = count.fit_transform(doc)
print(count.vocabulary_)
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')
from nltk.corpus import stopwords
#Classify the topics of movie set
df = pd.read_csv('movie_data.csv',encoding = 'utf-8')
count = CountVectorizer(stop_words = 'english',max_df=.15,max_features=5000)
X = count.fit_transform(df['review'].values)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 10, random_state = 0, learning_method='batch')
X_topics = lda.fit_transform(X)
n_top_words = 30
feature_names = count.get_feature_names()
for topic_idx,topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx+1))
    print(" ".join([feature_names[i] for i in topic.argsort()\
        [:-n_top_words-1:-1]]))
    
    
    
           
