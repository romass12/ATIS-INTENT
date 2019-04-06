
import importlib
import json
import spacy
import base64
import pickle,os,sys,time
import csv,spacy
import json,re
import os,pickle
import sklearn
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.utils import shuffle
import json,re
import numpy as np
import logging
import itertools
import spacy
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

logger = logging.getLogger(__name__)



class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size




def clean(word):
    '''This is for cleaning data before pickling data '''
    word = word.replace("b'",'')
    word = re.sub(r'\d+', '', word)
    word = re.sub('\s+', ' ', word).strip()
    word = word.replace('\r','')
    word = word.replace('\\r','')
    word = word.replace('\n','')
    word = word.replace('\\n','')
    word = word.replace("'",'')
    word = word.replace('\ufeff','')
    char_not_allowed = ['!','@','#','$','%','^','&','*','(',')','-','_','\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'']
    for ch in char_not_allowed:
        word=word.replace(ch,'')
    return word
def get_stopwords():
    stopword = open("./data/ATIS_STOPWORDS.txt", 'r')
    st = stopword.readlines()
    sto=[]
    for s in st:
        s = s.replace('\n','')
        sto.append(s)
    return sto

def pos_tagg(test_query,pattern):
    sentence = Sentence(test_query)
    model.predict(sentence)
    sentence_tagged=[sentence.to_tagged_string()]
    final_list=[]
    l1 = []
    if "<<NNP>>" in sentence_tagged[0]:
        sent_replace=[sentence_tagged[0].replace("<<NNP>>","*")]
        sent1=sent_replace[0]
        l1=re.findall(pattern,sent1)
        test_query = pos_remove(test_query,l1)
        

        
    return test_query
    
def pos_remove(sentence,list_of_nnp,list_of_day,list_of_dayperiod):
    for nnp in list_of_nnp:
        sentence = re.sub(r'\s\b%s\b\s'%nnp,' ',sentence)
    for day in list_of_day:
        sentence = re.sub(r'\s\b%s\b\s'%day,' weekday ',sentence)
    for day in list_of_dayperiod:
        sentence = re.sub(r'\s\b%s\b\s'%day,' timing ',sentence)
        
    return sentence
def remove_stopwords(label):
    labels=[]
    label=label.split()
    stopwords = get_stopwords()
    for word in label:
        if word not in stopwords:
            labels.append(lemmatizer.lemmatize(word))
    label = " ".join(str(w) for w in labels)
    return label
def remove_stopword(label,stopwords):
    labels=[]
    label=label.split()
    for word in label:
        if word not in stopwords:
            labels.append(word)
    label = " ".join(str(w) for w in labels)
    return label

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    if os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            return pickle.load(MacOSFile(f))

def slot_filling(test_query,pred,flight_no,airfare,airline):
    if pred == 'atis_flight':
        if len(re.findall(flight_no,test_query)) > 0:
            if len(re.findall(airfare,test_query)) > 0:
                return 'atis_flight#atis_airfare'
            if len(re.findall(airline,test_query)) > 0:
                return 'atis_flight_no#atis_airline'
            return 'atis_flight_no'
        if len(re.findall(airline,test_query)) > 0:
            return 'atis_flight#atis_airline'
    return pred
def ner_tagging(path_ner_data):
    list_of_nnp,list_of_day,list_of_rt,list_of_dayperiod=[],[],[],[]
    with open(path_ner_data, 'r',encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            row = list(filter(None, row))
            row =  row[:-1]
            rem = int((len(row)-1)/2)
            del row[rem]
            expression_length = len(row)
            list_of_nnp_indices = [i-int(expression_length/2) for i, x in enumerate(row) if x in ["B-fromloc.city_name","B-toloc.city_name"]]
            lis = list(map(row.__getitem__,list_of_nnp_indices))
            list_of_nnp.extend(lis)
            list_of_day_indices = [i-int(expression_length/2) for i, x in enumerate(row) if x in ["B-depart_date.day_name","B-arrive_date.day_name"]]
            lis = list(map(row.__getitem__,list_of_day_indices))
            list_of_day.extend(lis)
            list_of_rt_indices = [i-int(expression_length/2) for i, x in enumerate(row) if x in ["B-round_trip"]]
            lis = list(map(row.__getitem__,list_of_rt_indices))
            list_of_rt.extend(lis)
            list_of_dayperiod_indices = [i-int(expression_length/2) for i, x in enumerate(row) if x in ["B-depart_time.period_of_day"]]
            lis = list(map(row.__getitem__,list_of_dayperiod_indices))
            list_of_dayperiod.extend(lis)
        return list_of_nnp,list_of_day,list_of_rt,list_of_dayperiod

def training(training_data,list_of_nnp,list_of_day,list_of_dayperiod):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([pos_remove(i[0],list_of_nnp,list_of_day,list_of_dayperiod) for i in training_data])
    le = preprocessing.LabelEncoder()
    y_train = [i[1] for i in training_data]
    Y_train=le.fit_transform(y_train)

    X_train,y_train = shuffle(X_train, Y_train, random_state=42)
    clf = MultinomialNB(alpha=0.2)
    clf.fit(X_train, y_train)
    return vectorizer,clf,le
