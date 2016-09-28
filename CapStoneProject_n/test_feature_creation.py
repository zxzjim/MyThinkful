import numba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from IPython.display import display

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import wordnet as wn

import string
from scipy import stats
import multiprocessing
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

attributes_raw = pd.read_csv('attributes.csv')
product_desc_raw = pd.read_csv('product_descriptions.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')

test_merged = test_raw.merge(product_desc_raw, on='product_uid', how='left')

def remove_non_ascii(s):
    printable = set(string.printable)
    return filter(lambda x: x in printable, s)

def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def remove_stopwords(text):
    stops = stopwords.words('english')
    return [word for word in text if word not in stops]

def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)
def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words('english')]

def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)

def get_freq_in_text(text, word):
    freq = FreqDist(text)
    return freq[word]

def compare_synsets(synsets1, synsets2):
    comparisons = [syn1.path_similarity(syn2) for syn2 in synsets2 for syn1 in synsets1]
    comparisons = [v for v in comparisons if v is not None]
    return sum(comparisons)

def get_synsets(word):
    return wn.synsets(word)

def check_similarity_word_words(ref_word_synsets, words):
    synsets_of_all_words = [wn.synsets(word) for word in words]
#     print synsets_of_all_words
    sim_word_to_word  = [compare_synsets(ref_word_synsets, synsets) for synsets in synsets_of_all_words]
    return sum(sim_word_to_word)

def cal_similarities_mean(similarities_list):
    sims = [v for k,v in similarities_list.iteritems()]
    #drop na
    sims = np.array([e for e in sims if e != None])
    return sims.mean()

# core function to find the similarity between the search term and the target text
def find_search_similarity_title_desc(row,title_col_name,search_col_name, desc_col_name, mode):
#     print type(row)
    texts = [row[title_col_name],row[search_col_name],row[desc_col_name] ]
#     print texts
    
    sim_kw_title_mean_all = {}
    sim_kw_desc_mean_all = {}
    
    for kw in texts[1]:
#         print 'keyword: ', kw
        #get the synsets of the keyword
        kw_syn = wn.synsets(kw)
        #get the similarity matrix of kw:product_title
        sim_kw_title_mean = check_similarity_word_words(kw_syn, texts[0])
        #get the similarity matrix of kw:product_description
        sim_kw_desc_mean = check_similarity_word_words(kw_syn, texts[2])
    
        sim_kw_title_mean_all[kw] = sim_kw_title_mean
        sim_kw_desc_mean_all[kw] = sim_kw_desc_mean
    
    
    sim_title_mean_np = np.array(sim_kw_title_mean_all.values())
    sim_desc_mean_np = np.array(sim_kw_desc_mean_all.values())
    
    sim_title_mean_val = np.mean(sim_title_mean_np)
    sim_desc_mean_val = np.mean(sim_desc_mean_np)
    
#     print 'sim means: ', sim_title_mean_val, sim_desc_mean_val
    if mode =='title':
        print 'row id: ', row.id, ', return: ', sim_title_mean_val
        return sim_title_mean_val
    elif mode == 'desc':
        print 'row id: ', row.id, ', return: ', sim_desc_mean_val
        return sim_desc_mean_val
    else:
#         print 'row id: ', row.id, ', return: ', (sim_title_mean_val + sim_desc_mean_val)/2
        return (sim_title_mean_val + sim_desc_mean_val)/2

# wrapper functions so that the pool.map() can call the core function
def find_search_similarity_title(row_number):
    return find_search_similarity_title_desc(test_merged.iloc[row_number, :],'product_title','search_term', 'product_description', 'title')

def find_search_similarity_desc(row_number):
    return find_search_similarity_title_desc(test_merged.iloc[row_number, :],'product_title','search_term', 'product_description', 'desc')

def find_search_freq_title_desc(row,title_col_name,search_col_name, desc_col_name, mode):
#     print 'row id: ', row.id
    texts = [row[title_col_name],row[search_col_name],row[desc_col_name] ]
    
    if mode == 'title':
        counts = [get_freq_in_text(texts[0], kw) for kw in texts[1]]
#         print counts
        f = sum(counts)
    elif mode == 'desc':
        counts = [get_freq_in_text(texts[2], kw) for kw in texts[1]] 
        f = sum(counts)
    else:
        f = (sum([get_freq_in_text(texts[0], kw) for kw in texts[1]]) + sum([get_freq_in_text(texts[2], kw) for kw in texts[1]]))/2
#     print 'f: ', f
    return f

test_merged['product_title'] = test_merged['product_title'].apply(remove_non_ascii)
test_merged['product_description'] = test_merged['product_description'].apply(remove_non_ascii)
test_merged['search_term'] = test_merged['search_term'].apply(remove_non_ascii)

test_merged['product_title'] = test_merged['product_title'].apply(tokenize_text)
test_merged['product_description'] = test_merged['product_description'].apply(tokenize_text)
test_merged['search_term'] = test_merged['search_term'].apply(tokenize_text)

test_merged['product_title'] = test_merged['product_title'].apply(remove_stopwords)
test_merged['product_description'] = test_merged['product_description'].apply(remove_stopwords)
test_merged['search_term'] = test_merged['search_term'].apply(remove_stopwords)

test_merged['product_title'] = test_merged['product_title'].apply(get_words_stem)
test_merged['product_description'] = test_merged['product_description'].apply(get_words_stem)
test_merged['search_term'] = test_merged['search_term'].apply(get_words_stem)

test_merged['product_title_len'] = test_merged['product_title'].apply(len)
test_merged['product_description_len'] = test_merged['product_description'].apply(len)
test_merged['search_term_len'] = test_merged['search_term'].apply(len)

test_merged['freq_title_sum'] = test_merged.apply(lambda row: find_search_freq_title_desc(row, 'product_title','search_term', 'product_description', mode='title'), axis=1)
test_merged['freq_desc_sum'] = test_merged.apply(lambda row: find_search_freq_title_desc(row, 'product_title','search_term', 'product_description', mode='desc'), axis=1)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=32)
    test_merged['sim_title'] = pool.map(find_search_similarity_title, range(0, len(test_merged)))

    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=32)
    test_merged['sim_desc'] = pool.map(find_search_similarity_desc, range(0, len(test_merged)))
    
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on

test_merged['ratio_title'] = test_merged['freq_title_sum'] / test_merged['product_title_len']
test_merged['ratio_desc'] = test_merged['freq_desc_sum'] / test_merged['product_description_len']

test_merged.to_csv('test_merged_with_features.csv')