# -*- coding: utf-8 -*-
"""
Author ROMA JAIN
"""
import importlib
import warnings
import tensorflow as tf
import keras
from random import shuffle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,SpatialDropout1D
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.layers import Input, Dense, Dropout, Flatten,Activation,BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import sys
import pickle
import re,csv,os,json
import time
import codecs,argparse
from keras.preprocessing import text
from keras.optimizers import Adam
from utils import pickle_dump,remove_stopwords,ner_tagging,training

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
def set_keras_backend(backend):
    from keras import backend as K
    print(K.backend())

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")

ap = argparse.ArgumentParser()
ap.add_argument("-flag", "--flag", required=True,help="predict or train")



EMEDDING_FILE = 'glove.6B.100d.txt'
embed = 100
vector_size=64#100#500
epochs=12#35
batch_size=16
maxlen = 31


def sensitivity(y_true, y_pred):
    '''sensitivity as one metric for checking accuracy of the model (unbalanced classes)'''
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1score(y_true, y_pred):
    from keras import backend as K
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def load_pretrained(EMEDDING_FILE,t,vocab_size):
    embeddings_index = dict()
    words_not_found=[]
    f = open(EMEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    vocab_words = len(embeddings_index)
    print('found %s word vectors' % vocab_words)
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None  and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            pass
    return vocab_words,embedding_matrix,words_not_found
    

def get_XY(expressions,labels):
    lengt=[] 
    distint_words=[]
    for express in expressions:
        words = express.replace('\n','').split()
        leng = len(words)
        lengt.append(leng)
        for word in words:
            distint_words.append(word)
    
    maxlen = max(lengt)
    label_set=set(labels)
    
    tk = text.Tokenizer(oov_token='OOV')
    tk.fit_on_texts(expressions)
    
    distinct_words = set(distint_words)
    print("number of distinct.  ",len(distinct_words))
    X = tk.texts_to_sequences(expressions)
    label_encoder = LabelEncoder()
    integer_encode = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encode.reshape(len(integer_encode), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    return X,Y,maxlen,len(distinct_words),len(label_set),label_encoder,tk,integer_encode,distint_words,expressions,labels,onehot_encoder


def model_train(maxlen,X_train,y_train,embedding_matrix,embed,len_distinct,len_label,epochs,batch_size,weight,vector_size,X_test,Y_test,filepath):
    import tensorflow as tf
    from keras import backend as K
    set_keras_backend("tensorflow")
    early_stopping = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)
    model = Sequential()
    model.add(Embedding(len_distinct, 100, input_length=maxlen,weights=[embedding_matrix],trainable=True))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(64,return_sequences=False))) # was false#was 100 for latst data  for more_data its 200
    model.add(Dropout(0.1))#8 --> 0.2
    model.add(Dense(len_label, activation='softmax'))
    adam=keras.optimizers.Adam(lr=0.0002,beta_1=0.8, beta_2=0.99)#0.0001
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1score,sensitivity,precision])
    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=True,verbose=1,class_weight=weight,validation_data=(X_test, Y_test))
    return model,hist

def get_class_weights(y, smooth_factor):
    from collections import Counter 
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}



dir_path = os.path.dirname(os.path.realpath(__file__))
path_nlp_data = os.path.join(dir_path,'data/ATIS_TRAINING_DATA.csv')
path_ner_data = os.path.join(dir_path,'data/ATIS_NER.csv')
path_nlp_test_data = os.path.join(dir_path,'data/ATIS_TEST_DATA.csv')
pattern = re.compile(r'\w*\s["*"]')
flight_no = re.compile(r'flight number')
airline = re.compile(r'airline')
airfare = re.compile(r'flight fare')

list_of_nnp,list_of_day,list_of_rt,list_of_dayperiod=[],[],[],[]

with open(path_nlp_data, 'r',encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    training_data = [(remove_stopwords(row[0]),row[1]) for row in reader]
with open(path_nlp_test_data, 'r',encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    test_data = [(remove_stopwords(row[0]),row[1]) for row in reader]
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
    


def predict():
    maxlen = 31
    print("Loading model .....and predicting again")
    import tensorflow as tf
    from keras.models import model_from_json
    import gensim
    filep = 'LSTM_spatial_WOW11_{epoch:02d}-{val_acc:.2f}.h5'
    early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='max')
    early_stopping = keras.callbacks.ModelCheckpoint(filep, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=2)
    model = load_model('LSTM_spatial_WOW.h5',custom_objects={'sensitivity': sensitivity,'f1score':f1score,'precision':precision})
    with open('lstm_weightsWOW.pickle', 'rb') as handle:
        weight = pickle.load(handle)
    with open('lstm_tkWOW.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('lstm_labelWOW.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    with open('lstm_X_trainWOW.pickle', 'rb') as handle:
        X_train = pickle.load(handle)
    with open('lstm_y_trainWOW.pickle', 'rb') as handle:
        y_train = pickle.load(handle)
    with open('lstm_X_testWOW.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    with open('lstm_y_testWOW.pickle', 'rb') as handle:
        y_test = pickle.load(handle)
    count=0
    for i in test_data:
        test_query = remove_stopwords(i[0])
        print(test_query)
        if(len(test_query) > 0):
            seq = tokenizer.texts_to_sequences([test_query])
            X = sequence.pad_sequences(seq, maxlen=maxlen,padding='post')
            y_pred = model.predict(X)
            y_pred1 = np.argmax(y_pred,axis=1)
            classes = label_encoder.inverse_transform(y_pred1)[0]
            print(classes)
            if classes == i[1]:
                count+=1
    print("Accuracy on test data --> " ,(count/len(test_data))*100)
    
    print("DONE VALIDATING....")
   
def train():
    import tensorflow as tf
    X,y_train,maxlen,len_distinct,len_label,label_encoder,tk,labels,distint_words,expressions,label,onehot_encoder= get_XY([i[0] for i in training_data],[i[1] for i in training_data])
    len_distinct=1000#for added small talk!!!!#3045#2930#its 3600-more data
    len_vocab_words,embedding_matrix,words_not_found= load_pretrained(EMEDDING_FILE,tk,len_distinct)
    weight = get_class_weights(labels,0.1)
    X_train = sequence.pad_sequences(X, maxlen=maxlen,padding='post',truncating='post')
    X_test=X_train
    y_test=y_train
    from sklearn.utils import shuffle
    X_test, y_test = shuffle(X_test, y_test)
    with open('lstm_X_trainWOW.pickle', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_y_trainWOW.pickle', 'wb') as handle:
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_X_testWOW.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_y_testWOW.pickle', 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_tkWOW.pickle', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_labelWOW.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lstm_weightsWOW.pickle', 'wb') as handle:
        pickle.dump(weight, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    model,history = model_train(maxlen,X_train,y_train,embedding_matrix,embed,len_distinct,len_label,epochs,batch_size,weight,vector_size,X_test,y_test,filepath)
    model.save('LSTM_spatial_WOW.h5')
    print(history.history['acc'])
    print("validating ---------->")
    print(history.history['val_acc'])
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f1_score = history.history['f1score']
    val_f1score=history.history['val_f1score']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure(1)
    plt.show()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure(2)
    plt.show()
    plt.plot(epochs, loss, 'b', label='Training f1score')
    plt.plot(epochs, val_loss, 'r', label='Validation f1score')
    plt.title('Training and validation f1score')
    plt.legend()
    plt.figure(3)
    plt.show()



if __name__ == '__main__':
    args = vars(ap.parse_args())
    if args['flag'] == 'train':
        train()
    if args['flag'] == 'test':
        predict()
    else:
        print(" wrong arugment !! use either 'train' or 'test' ")




