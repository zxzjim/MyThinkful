
# coding: utf-8

# # Importing Libraries and Data

# In[1]:


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


# ## try to find the similarity among the search term and the description and product title

# ### experiment on the single row:

# In[50]:

texts = [train_merged.iloc[1, 2].lower(), train_merged.iloc[1, 3].lower(),
         train_merged.iloc[1, 5].lower()]
texts


# #### extract the text, tokenize the sentences and clean up the text

# In[51]:

from nltk.tokenize import word_tokenize, RegexpTokenizer


# In[52]:

# tokenize and remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
texts_t = [tokenizer.tokenize(t) for t in texts]


# In[53]:

# remove stopwords
from nltk.corpus import stopwords


def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words('english')]

texts_t = map(remove_stopwords, texts_t)


# In[54]:

# remove suffix of the words
from nltk.stem.wordnet import WordNetLemmatizer


def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)

texts_t = map(get_words_stem, texts_t)


# #### use freqdist() to check the frequencies of each word and compare it with the search term

# In[55]:

from nltk import FreqDist


# In[56]:

def get_freq_in_text(text, word):
    #     print text
    freq = FreqDist(text)
    return freq[word]


# In[57]:

texts_t


# In[58]:

for word in texts_t[1]:
    print word
    print 'freq in title: ', get_freq_in_text(texts_t[0], word)
    print 'freq in desc: ', get_freq_in_text(texts_t[2], word)


# #### use synsets module to check simlilarity

#  ##### method:
# * text_words
#     * word
#     * word
#     * word
#     * word
#         * synset1 <-loop through each synset
#         * synset2
#         * synset3
#
#  compared to:
#     * ref_word
#         * synset1
#         * synset2
#
#  find the max similarity between eg word:synset1 and ref_word:synset2
#
#  append this simliarity into word's syn_sims
#
#  find the max similarity between word:synset2 and the ref_word's synsets
#  ...
#  until each synset in the word has found the max similarity to the ref_word
#
#  then return the max value of the word's syn_sims list to represent the similarity of the word to the ref_word
#

# In[129]:

kw = texts_t[1][1]
kw


# In[130]:

kw_syn = wn.synsets(kw)
kw_syn


# In[131]:

def compare_synsets(synsets1, synsets2):
    comparisons = [syn1.path_similarity(syn2)
                   for syn2 in synsets2 for syn1 in synsets1]
    comparisons = [v for v in comparisons if v is not None]
    return sum(comparisons)


# In[132]:

def get_synsets(word):
    return wn.synsets(word)


# In[133]:

def check_similarity_word_words(ref_word_synsets, words):
    synsets_of_all_words = [wn.synsets(word) for word in words]
#     print synsets_of_all_words
    sim_word_to_word = [compare_synsets(
        ref_word_synsets, synsets) for synsets in synsets_of_all_words]
    return sum(sim_word_to_word)


# In[134]:

print kw, kw_syn, texts_t[0]
sim_kw_title = check_similarity_word_words(kw_syn, texts_t[0])
kw, sim_kw_title


# * check similarity between search key word and the product description

# In[156]:

sim_kw_desc = check_similarity_word_list(kw_syn, texts_t[2])
kw, sim_kw_desc


# * mean values of the similarities

# In[157]:

def cal_similarities_mean(similarities_list):
    sims = [v for k, v in similarities_list.iteritems()]
    # drop na
    sims = np.array([e for e in sims if e != None])
    return sims.mean()


# In[158]:

sim_kw_title_mean = cal_similarities_mean(sim_kw_title)
print sim_kw_title_mean

sim_kw_desc_mean = cal_similarities_mean(sim_kw_desc)
print sim_kw_desc_mean


# ### run on all rows

# In[155]:

def remove_stopwords(text):
    return np.array([word for word in text if word not in stopwords.words('english')])


def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)


def compare_synsets(synsets1, synsets2):
    comparisons = [syn1.path_similarity(syn2)
                   for syn2 in synsets2 for syn1 in synsets1]
    comparisons = [v for v in comparisons if v is not None]
    if len(comparisons) > 0:
        return max(comparisons)
    else:
        return 0


def get_synsets(word):
    return wn.synsets(word)


def check_similarity_word_words(ref_word_synsets, words):
    synsets_of_all_words = [wn.synsets(word) for word in words]
#     print synsets_of_all_words
    sim_word_to_word = [compare_synsets(
        ref_word_synsets, synsets) for synsets in synsets_of_all_words]
    return max(sim_word_to_word)


# In[156]:

def find_search_similarity_title_desc(row, title_col_name, search_col_name, desc_col_name, mode):
    print 'row id: ', row.id
    texts = np.array(
        [row[title_col_name], row[search_col_name], row[desc_col_name]])
#     print texts

    # tokenize and remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    texts_t = np.array([tokenizer.tokenize(t) for t in texts])

    # remove stopwords
    texts_t = map(remove_stopwords, texts_t)

    # remove suffix of the words
    texts_t = map(get_words_stem, texts_t)

    sim_kw_title_mean_all = {}
    sim_kw_desc_mean_all = {}

    for kw in texts_t[1]:
        print 'keyword: ', kw
        # get the synsets of the keyword
        kw_syn = wn.synsets(kw)
        # get the similarity matrix of kw:product_title
        sim_kw_title_mean = check_similarity_word_words(kw_syn, texts_t[0])
        # get the similarity matrix of kw:product_description
        sim_kw_desc_mean = check_similarity_word_words(kw_syn, texts_t[2])

        sim_kw_title_mean_all[kw] = sim_kw_title_mean
        sim_kw_desc_mean_all[kw] = sim_kw_desc_mean

    sim_title_mean_np = np.array(sim_kw_title_mean_all.values())
    sim_desc_mean_np = np.array(sim_kw_desc_mean_all.values())

    sim_title_mean_val = np.mean(sim_title_mean_np)
    sim_desc_mean_val = np.mean(sim_desc_mean_np)

    print 'sim means: ', sim_title_mean_val, sim_desc_mean_val
    if mode == 'title':
        print 'return: ', sim_title_mean_val
        return sim_title_mean_val
    elif mode == 'desc':
        print 'return: ', sim_desc_mean_val
        return sim_desc_mean_val
    elif mode == 'avg':
        print 'return: ', (sim_title_mean_val + sim_desc_mean_val) / 2
        return (sim_title_mean_val + sim_desc_mean_val) / 2


# In[157]:

train_merged_sub = train_merged.iloc[0:10, :]
train_merged_sub.info()


# In[158]:

train_merged['sim_title'] = train_merged.apply(lambda row: find_search_similarity_title_desc(
    row, 'product_title', 'search_term', 'product_description', 'title'), axis=1)


# In[19]:

train_merged['sim_desc'] = train_merged.apply(lambda row: find_search_similarity_title_desc(
    row, 'product_title', 'search_term', 'product_description', 'desc'), axis=1)


# In[22]:

train_merged.to_csv('train_merged_with_2_new_features')


# In[ ]:

# multiple linear regression
# SVM
# random forest

# vis: scatter matrix
#     confusion matrix
#     roc curve
# word cloud:
# https://github.com/shubhabrataroy/Thinkful/blob/master/Curriculum/SetNoteBook.ipynb
