
# coding: utf-8

# # Importing Libraries and Data

# In[1]:

# get_ipython().magic(u'matplotlib inline')
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

import multiprocessing


# In[2]:

attributes_raw = pd.read_csv('attributes.csv')
product_desc_raw = pd.read_csv('product_descriptions.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')


# In[3]:

attributes_raw.head(20)


# In[4]:

attributes_raw.info()


# In[5]:

product_desc_raw.head(20)


# In[6]:

product_desc_raw.info()


# In[7]:

train_raw.head(20)


# In[8]:

train_raw.info()


# In[9]:

test_raw.head(20)


# # Data Wrangling

# In[10]:

train_merged = train_raw.merge(product_desc_raw, on='product_uid', how='left')
train_merged.tail(20)


# In[11]:

def remove_non_ascii(s):
    printable = set(string.printable)
    return filter(lambda x: x in printable, s)


# In[12]:

train_merged['product_title'] = train_merged[
    'product_title'].apply(remove_non_ascii)
train_merged['product_description'] = train_merged[
    'product_description'].apply(remove_non_ascii)
train_merged['search_term'] = train_merged[
    'search_term'].apply(remove_non_ascii)


# In[13]:

df = train_merged[train_merged['id'] == 1060]
print df


# In[14]:

def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)


def remove_stopwords(text):
    stops = stopwords.words('english')
    return [word for word in text if word not in stops]


def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)


# In[15]:

train_merged['product_title'] = train_merged[
    'product_title'].apply(tokenize_text)
train_merged['product_description'] = train_merged[
    'product_description'].apply(tokenize_text)
train_merged['search_term'] = train_merged['search_term'].apply(tokenize_text)


# In[16]:

train_merged.head()


# In[17]:

train_merged['product_title'] = train_merged[
    'product_title'].apply(remove_stopwords)
train_merged['product_description'] = train_merged[
    'product_description'].apply(remove_stopwords)
train_merged['search_term'] = train_merged[
    'search_term'].apply(remove_stopwords)


# In[18]:

train_merged.head()


# In[19]:

train_merged['product_title'] = train_merged[
    'product_title'].apply(get_words_stem)
train_merged['product_description'] = train_merged[
    'product_description'].apply(get_words_stem)
train_merged['search_term'] = train_merged['search_term'].apply(get_words_stem)


# In[20]:

train_merged.head()


# In[38]:

train_merged['product_title_len'] = train_merged['product_title'].apply(len)
train_merged['product_description_len'] = train_merged[
    'product_description'].apply(len)
train_merged['search_term_len'] = train_merged['search_term'].apply(len)


# In[39]:

train_merged.head()


# ## try to find the similarity among the search term and the description and product title


# ### run on all rows

# In[40]:

def get_freq_in_text(text, word):
    #     print text
    freq = FreqDist(text)
    return freq[word]


def compare_synsets(synsets1, synsets2):
    comparisons = [syn1.path_similarity(syn2)
                   for syn2 in synsets2 for syn1 in synsets1]
    comparisons = [v for v in comparisons if v is not None]
    if len(comparisons) > 0:
        return max(comparisons)
    else:
        return 0


def check_similarity_word_words(ref_word_synsets, words):
    synsets_of_all_words = [wn.synsets(word) for word in words]
#     print synsets_of_all_words
    sim_word_to_word = [compare_synsets(
        ref_word_synsets, synsets) for synsets in synsets_of_all_words]
    sim_word_to_word = [v for v in sim_word_to_word if v > 0]
    if len(sim_word_to_word) > 0:
        return np.mean(np.array(sim_word_to_word))
    else:
        return 0


# In[41]:

# core function to find the similarity between the search term and the
# target text
def find_search_similarity_title_desc(row, title_col_name, search_col_name, desc_col_name, mode):
    #     print type(row)
    texts = [row[title_col_name], row[search_col_name], row[desc_col_name]]
#     print texts

    sim_kw_title_mean_all = {}
    sim_kw_desc_mean_all = {}

    for kw in texts[1]:
        #         print 'keyword: ', kw
        # get the synsets of the keyword
        kw_syn = wn.synsets(kw)
        # get the similarity matrix of kw:product_title
        sim_kw_title_mean = check_similarity_word_words(kw_syn, texts[0])
        # get the similarity matrix of kw:product_description
        sim_kw_desc_mean = check_similarity_word_words(kw_syn, texts[2])

        sim_kw_title_mean_all[kw] = sim_kw_title_mean
        sim_kw_desc_mean_all[kw] = sim_kw_desc_mean

    sim_title_mean_np = np.array(sim_kw_title_mean_all.values())
    sim_desc_mean_np = np.array(sim_kw_desc_mean_all.values())

    sim_title_mean_val = np.mean(sim_title_mean_np)
    sim_desc_mean_val = np.mean(sim_desc_mean_np)

    print 'sim means: ', sim_title_mean_val, sim_desc_mean_val
    if mode == 'title':
        print 'row id: ', row.id, ', return: ', sim_title_mean_val
        return sim_title_mean_val
    elif mode == 'desc':
        print 'row id: ', row.id, ', return: ', sim_desc_mean_val
        return sim_desc_mean_val
    else:
        print 'row id: ', row.id, ', return: ', (sim_title_mean_val + sim_desc_mean_val) / 2
        return (sim_title_mean_val + sim_desc_mean_val) / 2


# In[42]:

# wrapper functions so that the pool.map() can call the core function
def find_search_similarity_title(row_number):
    return find_search_similarity_title_desc(train_merged_sub.iloc[row_number, :], 'product_title', 'search_term', 'product_description', 'title')


def find_search_similarity_desc(row_number):
    return find_search_similarity_title_desc(train_merged_sub.iloc[row_number, :], 'product_title', 'search_term', 'product_description', 'desc')


# In[43]:

def find_search_freq_title_desc(row, title_col_name, search_col_name, desc_col_name, mode):
    print 'row id: ', row.id
    texts = [row[title_col_name], row[search_col_name], row[desc_col_name]]

    if mode == 'title':
        counts = [get_freq_in_text(texts[0], kw) for kw in texts[1]]
#         print counts
        f = sum(counts)
    elif mode == 'desc':
        counts = [get_freq_in_text(texts[2], kw) for kw in texts[1]]
        f = sum(counts)
    else:
        f = (sum([get_freq_in_text(texts[0], kw) for kw in texts[1]]) +
             sum([get_freq_in_text(texts[2], kw) for kw in texts[1]])) / 2
#     print 'f: ', f
    return f


# In[44]:

train_merged_sub = train_merged.iloc[0:100, :]
train_merged_sub.info()


# In[45]:

train_merged_sub['freq_title_sum'] = train_merged_sub.apply(lambda row: find_search_freq_title_desc(row, 'product_title', 'search_term', 'product_description', mode='title'), axis=1)
train_merged_sub['freq_desc_sum'] = train_merged_sub.apply(lambda row: find_search_freq_title_desc(row, 'product_title', 'search_term', 'product_description', mode='desc'), axis=1)


# In[46]:

train_merged_sub.head()


# In[51]:

get_ipython().run_cell_magic(u'time', u'',
                             u"if __name__ == '__main__':\n    pool = multiprocessing.Pool(processes=16)\n    train_merged_sub['sim_title'] = pool.map(find_search_similarity_title, range(0, len(train_merged_sub)))\n\n    pool.close() #we are not adding any more processes\n    pool.join() #tell it to wait until all threads are done before going on")


# In[48]:

train_merged_sub.head()


# In[49]:

get_ipython().run_cell_magic(u'time', u'',
                             u"if __name__ == '__main__':\n    pool = multiprocessing.Pool(processes=16)\n    train_merged_sub['sim_desc'] = pool.map(find_search_similarity_desc, range(0, len(train_merged_sub)))\n    \n    pool.close() #we are not adding any more processes\n    pool.join() #tell it to wait until all threads are done before going on")


# In[50]:

train_merged_sub.tail(20)


# In[52]:

train_merged_sub['ratio_title'] = train_merged_sub[
    'freq_title_sum'] / train_merged_sub['product_title_len']
train_merged_sub['ratio_desc'] = train_merged_sub[
    'freq_desc_sum'] / train_merged_sub['product_description_len']


# In[53]:

print train_merged_sub.head()

train_merged_sub.to_csv('train_merged_sub.csv')
# * multiple linear regression
# * SVM
# * random forest
#
# * vis: scatter matrix
#    * confusion matrix
#    * roc curve
#    * word cloud: https://github.com/shubhabrataroy/Thinkful/blob/master/Curriculum/SetNoteBook.ipynb
#

# In[ ]:
