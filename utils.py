
import importlib
import json
import spacy
import base64
import pickle,os,sys,time
from functools import wraps
import csv,spacy
import json,re
import os,pickle
import sklearn
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import preprocessing
from sklearn.utils import shuffle
import json,re
import numpy as np
import logging
import itertools
import spacy
from nltk.stem import WordNetLemmatizer 
from flair.models import SequenceTagger
from flair.data import Sentence

spacy_model=spacy.load('en')

model = SequenceTagger.load('pos')
  
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

def find_ngrams(input_list, n):
    """Create ngrams for input list
        @param: input_list The list to separate in ngrams
        @param: n Number of ngrams to find"""
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    """ To create the tokens from the ngrams with #token# style
        @param: text String to transform
        
        return final_tokens list of ngrams for words with hashes at the beginning and end"""
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        #if unhashed_token not in stopwords:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens+=[hashed_token]
        #final_tokens += [''.join(gram) for gram in list(find_ngrams(list(hashed_token), 3))]
        #print(final_tokens)
    return final_tokens
    
class ConfigLoading():
    def __init__(self,dir_path,config,path_nlp_data,path_nlp_models):
        self.config = config
        self.score_srk_en = config['data']['score_srk_en']
        self.ratio_srk_en = config['data']['ratio_srk_en']
        self.path_nlp_data = path_nlp_data
        self.path_nlp_models = path_nlp_models
        self.nlp_model = self.load_models(os.path.join(self.path_nlp_models,config['models']['NLP']['NLP_MODEL']),'spacy')
        self.en_stopwords_path = os.path.join(self.path_nlp_data,config['data']['stopwords_en'])
        self.stopwords_en = self.load_data(self.en_stopwords_path)

        
        self.label_Enco = self.load_models(os.path.join(self.path_nlp_models,config['models']['label_encoder']),'pickle')
        self.classifier = self.load_models(os.path.join(self.path_nlp_models,config['models']['classifier']),'pickle')
        self.vectorizer = self.load_models(os.path.join(self.path_nlp_models,config['models']['vectorizer']),'pickle')
        self.NLP_MODELS= list(self.config['models']['NLP'].keys())
        self.NLP_DATA= list(self.config['data'].keys())
        self.nlp_path = os.path.join(self.path_nlp_data,config['data']['unique_en'])
        self.unique  = self.load_models(self.nlp_path,'pickle')
        self.responses_dict  = self.load_data(os.path.join(self.path_nlp_data,config['data']['responses']))
        

    def load_data(self,path):
        logger.debug("loading data from : {}".format(path))
        if(path.endswith('json')):
            with open(path) as f:
                data = json.loads(f.read())
            return data
        with open(path,'r') as f:
            data = f.readlines()
        sto=[]
        for s in data:
        	s = s.replace('\n','')
        	sto.append(s)
        return sto
    def load_models(self,path,flag):
        logger.debug("loading from {} with type {}".format(path, flag))
        if flag == 'spacy':
            return spacy.load(path)
        if flag == 'pickle':
            try:
                return pickle.load(open(path,'rb'))
            except Exception as e:
                logger.debug("cannot load this file because {}".format(e))
                try:
                    return pickle_load(path)
                except:
                    logger.warning("this pickle already has a lot of data that even after doing buffered reading on file it is still giving error")

def get_class_weights(y_train, smooth_factor):
    from collections import Counter 
    counter = Counter(y_train)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    weight = {cls: float(count / majority) for cls, count in counter.items()}
    weights=[]
    classs=[]
    for y in y_train:
        if y not in classs:
            classs.append(y)
            weights.append(weight[y])
    return weights

def similarity(sentence,list_of_nnp):
    for word in sentence.split():
        token = spacy_model(word)
        for nnp in list(set(list_of_nnp)):
            if token.similarity(spacy_model(nnp)) > 0.8:
                print(word)
                sentence = re.sub(r'\s\b%s\b\s'%word,' ',sentence)
                break
    return sentence



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
def uploaded_data(data,nlp_model):
    '''This method aids in extracting data from uploaded file using POS model'''

    entities=[]
    final_data=[]
    vocab={}
    final_data = [sentence.split(' ') for sentence in data]  
    final_data = list(itertools.chain.from_iterable(final_data))
    logger.debug("length of vocabulary words -->{}".format(len(final_data)))
    
    return final_data

def change_existing_dataset(data,nlp_model,list_of_stop,path):
    '''This is for storing vocab words in form of dictionary and dumping pos words extracted '''
    words=[]
    # ======================================= FOR KLEIN WITH TWISTED -- _file.value ========================================
    words = uploaded_data(data,nlp_model)
    if not words:
        return False
    dict_words={}
    for wo in words:
        if (wo not in list_of_stop):
            dict_words[wo] = nlp_model(wo)
    pickle_dump(dict_words,path)
    logger.info('updated new vocabulary --->')
    return True