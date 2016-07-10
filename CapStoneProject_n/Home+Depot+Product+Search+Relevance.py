
# coding: utf-8

# # Importing Libraries and Data

# In[2]:

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

get_ipython().magic(u'matplotlib inline')


# In[3]:

attributes_raw = pd.read_csv('attributes.csv')
product_desc_raw = pd.read_csv('product_descriptions.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')


# In[4]:

attributes_raw.head(20)


# In[5]:

attributes_raw.info()


# In[6]:

product_desc_raw.head(20)


# In[7]:

product_desc_raw.info()


# In[8]:

train_raw.head(20)


# In[9]:

train_raw.info()


# In[10]:

test_raw.head(20)


# # Data Wrangling

# In[11]:

train_merged = train_raw.merge(product_desc_raw, on='product_uid', how='left')
train_merged.tail(20)


# In[12]:

def remove_non_ascii(s):
    printable = set(string.printable)
    return filter(lambda x: x in printable, s)


# In[13]:

train_merged['product_title'] = train_merged['product_title'].apply(remove_non_ascii)
train_merged['product_description'] = train_merged['product_description'].apply(remove_non_ascii)
train_merged['search_term'] = train_merged['search_term'].apply(remove_non_ascii)


# In[14]:

df = train_merged[train_merged['id']==1060]
print df


# ## try to find the similarity among the search term and the description and product title

# #### use freqdist() to check the frequencies of each word and compare it with the search term

# In[18]:

from nltk import FreqDist


# In[19]:

def get_freq_in_text(text, word):
#     print text
    freq = FreqDist(text)
    return freq[word]


# In[20]:

# for word in texts_t[1]:
#     print word
#     print 'freq in title: ', get_freq_in_text(texts_t[0], word)
#     print 'freq in desc: ', get_freq_in_text(texts_t[2], word)


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

# ### run on all rows

# In[21]:

def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words('english')]

def get_words_stem(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return map(lemmatizer.lemmatize, tokenized_text)

def check_similarity_word_words(word_syn, words):
    result = {}
    # for each word in the text
    for w in words:
        syn_sims = []
        if len(wn.synsets(w)) > 0 :
#             print 'target word & synsets: ', w, wn.synsets(w)
            # for each synonym in the synsets of the word
            for syn in wn.synsets(w):
#                 print syn, [ref_syn.path_similarity(syn) for ref_syn in word_syn]
                comparisons = [ref_syn.path_similarity(syn) for ref_syn in word_syn]
                # get the highest similarity between the synonyms of the reference word and that of the target word
                if len(comparisons) > 0:
                    sim = max(comparisons)
                else:
                    sim = 0
#                 print 'sim:', sim
                syn_sims.append(sim)
#             print w, syn_sims    
            result[w] = max(syn_sims)
    return result

def cal_similarities_mean(similarities_list):
    sims = [v for k,v in similarities_list.iteritems()]
    #drop na
    sims = np.array([e for e in sims if e != None])
    return sims.mean()



# In[22]:

def find_search_similarity_title_desc(row,title_col_name,search_col_name, desc_col_name, mode):
    print row.id
    texts = [row[title_col_name],row[search_col_name],row[desc_col_name] ]
#     print texts

    #tokenize and remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    texts_t = [tokenizer.tokenize(t) for t in texts]
    
    #remove stopwords
    texts_t = map(remove_stopwords, texts_t)
    
    #remove suffix of the words
    texts_t = map(get_words_stem, texts_t)
    
    sim_kw_title_mean_all = {}
    sim_kw_desc_mean_all = {}
    
    for kw in texts_t[1]:
        print kw
        #get the synsets of the keyword
        kw_syn = wn.synsets(kw)
        #get the similarity matrix of kw:product_title
        sim_kw_title = check_similarity_word_words(kw_syn, texts_t[0])
        #get the similarity matrix of kw:product_description
        sim_kw_desc = check_similarity_word_words(kw_syn, texts_t[2])
        #calculate the mean similarities in the kw:product_title similarity matrix
        sim_kw_title_mean =cal_similarities_mean(sim_kw_title)
        #calculate the mean similarities in the kw:product_description similarity matrix
        sim_kw_desc_mean = cal_similarities_mean(sim_kw_desc)
    
        sim_kw_title_mean_all[kw] = sim_kw_title_mean
        sim_kw_desc_mean_all[kw] = sim_kw_desc_mean
    
    sim_title_mean_np = np.array(sim_kw_title_mean_all.values())
    sim_desc_mean_np = np.array(sim_kw_desc_mean_all.values())
    
    sim_title_mean_val = np.mean(sim_title_mean_np)
    sim_desc_mean_val = np.mean(sim_desc_mean_np)
    
    print sim_title_mean_val, sim_desc_mean_val
    if mode =='title':
        print 'return: ', sim_title_mean_val
        return sim_title_mean_val
    elif mode == 'desc':
        print 'return: ', sim_desc_mean_val
        return sim_desc_mean_val
    elif mode == 'avg':
        print 'return: ', (sim_title_mean_val + sim_desc_mean_val)/2
        return (sim_title_mean_val + sim_desc_mean_val)/2


# In[ ]:

train_merged['sim_title'] = train_merged.apply(lambda row: find_search_similarity_title_desc(row, 'product_title','search_term', 'product_description', 'title'), axis=1)
train_merged['sim_desc'] = train_merged.apply(lambda row: find_search_similarity_title_desc(row, 'product_title','search_term', 'product_description', 'desc'), axis=1)
train_merged.to_csv('train_merged_f.csv')


# In[203]:

train_merged.head()


# In[ ]:

multiple linear regression
SVM
random forest

vis: scatter matrix
    confusion matrix
    roc curve
    word cloud: https://github.com/shubhabrataroy/Thinkful/blob/master/Curriculum/SetNoteBook.ipynb

