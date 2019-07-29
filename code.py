import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for part-of-speech tagging
from nltk import pos_tag

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# BeautifulSoup libraray
#from bs4 import BeautifulSoup 

import re # regex

#model_selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.metrics import classification_report
#from mlxtend.plotting import plot_confusion_matrix

#preprocessing scikit
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
 
#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input,CuDNNLSTM,LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence

#gensim w2v
#word2vec
from gensim.models import Word2Vec
#import nltk
#nltk.download('punkt')
import nltk.data
import gensim

data=pd.read_csv(r'/Users/shubhamkumar/Downloads/train.csv')
data_copy=data.copy()
data_copy=data_copy.drop(['id', 'App Version Code','App Version Name','Review Title'], axis=1)
data_copy=data_copy.dropna()
data_copy.drop_duplicates(subset=['Star Rating','Review Text'],keep='first',inplace=True)

def clean_reviews(review):  
    review_text = re.sub("[^a-zA-Z]"," ",review)
    word_tokens= review_text.lower().split()
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))     
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]
    
    cleaned_review=" ".join(word_tokens)
    return cleaned_review

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
sum=0
for reviews in data_copy['Review Text']:
  sents=tokenizer.tokenize(reviews)
  sum+=len(sents)
  for sent in sents:
    cleaned_sent=clean_reviews(sent)
    sentences.append(cleaned_sent.split()) 
print(sum)
print(len(sentences)) 

word_to_vector=gensim.models.Word2Vec(sentences=sentences,size=300,window=10,min_count=1)

word_to_vector.train(sentences,epochs=10,total_examples=len(sentences))

vocab=word_to_vector.wv.vocab
print("The total number of words are : ",len(vocab))
vocab=list(vocab.keys())

#word to vector 
word_vec_dict={}
for word in vocab:
  word_vec_dict[word]=word_to_vector.wv.get_vector(word)
print("The no of key-value pairs : ",len(word_vec_dict))

#add a new column 
data_copy['clean_review']=data_copy['Review Text'].apply(clean_reviews)

#get the maximum length of words
maxi=-1
data_copy['clean_review'].dropna(inplace=True)
for i,rev in enumerate(data_copy['clean_review']):
  tokens=rev.split()
  if(len(tokens)>maxi):
    maxi=len(tokens)
print(maxi)

tok = Tokenizer()
tok.fit_on_texts(data_copy['clean_review'])
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(data_copy['clean_review'])

pad_rev= pad_sequences(encd_rev, maxlen=maxi+1, padding='post')

#embedded matrix
embed_dim=300
embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word_vec_dict.get(word)
  if embed_vector is not None:  
    embed_matrix[i]=embed_vector

Y=keras.utils.to_categorical(data_copy['Star Rating'])
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)

from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout
model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length=maxi+1,embeddings_initializer=Constant(embed_matrix)))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(6,activation='sigmoid'))

print(model.summary())

model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])

epochs=7
batch_size=64

model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

loss, accuracy = model.evaluate(x_train,y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print('loss: %f' %(loss*100))

test_data=test.copy()
test_data=test_data.drop([ 'App Version Code','App Version Name','Review Title'], axis=1)
test_data=test_data.dropna()

test
sentences_for_test=[]
sum=0
for reviews in test_data['Review Text']:
  sents=tokenizer.tokenize(reviews.strip())
  #print(sents)
  sum+=len(sents)
  for sent in sents:
    cleaned_sent=clean_reviews(sent)
    sentences_for_test.append(cleaned_sent.split()) # can use word_tokenize also.
print(sum)
print(len(sentences_for_test))

word_to_vector_test=gensim.models.Word2Vec(sentences=sentences_for_test,size=300,window=10,min_count=1)

word_to_vector_test.train(sentences_for_test,epochs=10,total_examples=len(sentences_for_test))


vocab_test=word_to_vector_test.wv.vocab
print("The total number of words are : ",len(vocab_test))

vocab_test=list(vocab_test.keys())

word_vec_dict_test={}
for word in vocab_test:
  word_vec_dict_test[word]=word_to_vector_test.wv.get_vector(word)
print("The no of key-value pairs : ",len(word_vec_dict_test))

test_data['clean_review']=test_data['Review Text'].apply(clean_reviews)

tok = Tokenizer()
tok.fit_on_texts(test_data['clean_review'])
vocab_size_test = len(tok.word_index) + 1
encd_rev_test = tok.texts_to_sequences(test_data['clean_review'])

pad_rev_test= pad_sequences(encd_rev_test, maxlen=maxi+1, padding='post')

embed_dim=300
embed_matrix=np.zeros(shape=(vocab_size_test,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word_vec_dict_test.get(word)
  if embed_vector is not None:  
    embed_matrix[i]=embed_vector

x=model.predict(pad_rev_test)
pred=np.argmax(x,axis=1)

out = pd.DataFrame(list(zip(test_data['id'],pred)), 
               columns =['id', 'rating'])

export_csv = df.to_csv (r'\Users\shubhamkumar\submission2.csv', index = None, header=True)