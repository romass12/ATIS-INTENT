# coding: utf-8
'''author Roma Jain'''
# coding: utf-8

import csv,re
import os,pickle
import sklearn
from tqdm import tqdm
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import preprocessing
from sklearn.utils import shuffle
import argparse
import logging 

from utils import pickle_dump,remove_stopwords,ner_tagging,training,pickle_load,pos_remove,slot_filling
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-flag", "--flag", required=True,help="predict or train")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename = "logs_info.log",level=logging.DEBUG,format="%(asctime)s:%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)
logger.info('now loading configs and config class')

'''Config loading '''
dir_path = os.path.dirname(os.path.realpath(__file__))
path_data = os.path.join(dir_path,'data')
path_nlp_data = os.path.join(dir_path,'data/ATIS_TRAINING_DATA.csv')
path_ner_data = os.path.join(dir_path,'data/ATIS_NER.csv')
path_nlp_test_data = os.path.join(dir_path,'data/ATIS_TEST_DATA.csv')
path_models = os.path.join(dir_path,'models')
pattern = re.compile(r'\w*\s["*"]')
flight_no = re.compile(r'flight number')
airline = re.compile(r'airline')
airfare = re.compile(r'flight fare')
label_enco = os.path.join(path_models,'naive_bayes_ATIS5' + '.pickle')
semhash_feature =os.path.join(path_models,'naive_bayes_ATISfeature5' + '.pickle')
model_NB =  os.path.join(path_models,'naive_bayes_ATIS_MODEL5'  +  '.pickle')
nnp = os.path.join(path_data,'nnp_list'  +  '.pickle')
day =os.path.join(path_data, 'day_list' +  '.pickle')
rt = os.path.join(path_data,'rt' +  '.pickle')
dayperiod = os.path.join(path_data,'dayperiod' + '.pickle')

'''Training data loaded'''
def train():
	logger.info("loading training data --->")
	with open(path_nlp_data, 'r',encoding="utf-8-sig") as f:
	    reader = csv.reader(f)
	    training_data = [(remove_stopwords(row[0]),row[1]) for row in reader]
	'''NER TAGS extracted using BIOS TAGS'''
	list_of_nnp,list_of_day,list_of_rt,list_of_dayperiod = ner_tagging(path_ner_data)
	'''TRAINING USING NB + CV'''
	vectorizer,clf,le = training(training_data,list_of_nnp,list_of_day,list_of_dayperiod)
	'''SAVING MODELS AND DATA '''
	logger.info("trained NB on training data and saving models --->")
	pickle_dump(clf, model_NB)
	pickle_dump(le, label_enco)
	pickle_dump(vectorizer, semhash_feature)
	pickle_dump(list_of_nnp,nnp)
	pickle_dump(list_of_day,day)
	pickle_dump(list_of_rt,rt)
	pickle_dump(list_of_dayperiod,dayperiod)

def test():
	logger.info("loading models --->")
	le = pickle_load(label_enco)
	vectorizer = pickle_load(semhash_feature)
	clf = pickle_load(model_NB)
	list_of_nnp = pickle_load(nnp)
	list_of_day = pickle_load(day)
	list_of_dayperiod = pickle_load(dayperiod)
	logger.info("loading testing data --->")
	with open(path_nlp_test_data, 'r',encoding="utf-8-sig") as f:
		reader = csv.reader(f)
		test_data = [(remove_stopwords(row[0]),row[1]) for row in reader]
	count=0
	for i in tqdm(test_data):
	    #removing inconstant words like proper nouns words and weekday names and period of day since our classification should be independent of them
	    test_query = pos_remove(i[0],list_of_nnp,list_of_day,list_of_dayperiod)
	    test_query = remove_stopwords(test_query)
	    if(len(test_query) > 0):
	        test_quer = vectorizer.transform([test_query])
	        pred = clf.predict(test_quer)
	        pro = clf.predict_proba(test_quer)[0]
	        PRO = pro[np.argmax(pro)]
	        pred =  le.inverse_transform(pred)[0]
	        #slot filling modifies the intent according to certain keywords 
	        pred = slot_filling(test_query,pred,flight_no,airfare,airline)
	        if pred == i[1]:
	            count+=1
	
	return count,len(test_data)


if __name__ == '__main__':
	args = vars(ap.parse_args())
	logger.debug("Args are {}".format(args))
	if args['flag'] == 'train':
		train()
	if args['flag'] == 'test':
		acc,len_test = test()
		logger.debug("Accuracy on test data -->{}".format(acc/len_test)*100)
	else:
		logger.warning(" wrong arugment !! use either 'train' or 'test' ")
