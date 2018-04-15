
# coding: utf-8

# # Introduction

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

import json
import ucto
import pickle
import sys
import os.path

_show_graphics = True

if __name__ == '__main__' and '__file__' in globals():
    # running in console
    _show_graphics = False


# In[2]:


from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

                      
cm = confusion_matrix(['BEL', 'BEL', 'DUT', 'DUT', 'BEL', 'DUT', 'DUT', 'DUT', 'DUT', 'DUT'],  # golden truth
                      ['BEL', 'DUT', 'DUT', 'BEL', 'DUT', 'DUT', 'BEL', 'BEL', 'DUT', 'DUT'])  # our predictions

labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(["BEL", "DUT"])
print("BEL", labelEncoder.transform(["BEL"]))
print("DUT", labelEncoder.transform(["DUT"]))

if _show_graphics:
    plt.figure()
    plot_confusion_matrix(cm, classes=labelEncoder.classes_, title="My first confusion matrix")


# In[3]:


ucto_config = "tokconfig-nld"
tokeniser = ucto.Tokenizer(ucto_config, sentenceperlineinput=True, sentencedetection=False, paragraphdetection=False)

def read_data(file):
    text = {}
    with open(file) as f:
        for line in tqdm(f):
            sentence, language = line.strip().split("\t")
            tokeniser.process(sentence)

            if language not in text:
                text[language] = []

            current_line = []
            for token in tokeniser:
                current_line.append(str(token))
                if token.isendofsentence():
                    #print(current_line)
                    text[language].append(" ".join(current_line))
                    current_line = []
    return text


# In[4]:


# First the development set
try:
    with open('data/dev.txt.pickle', 'rb') as f:
        _l_dev_text = pickle.load(f)
        print("Done reading development set from pickle.")
except IOError:
    _l_dev_text = read_data('data/dev.txt')
    print("Done tokenising development set.")
    with open('data/dev.txt.pickle', 'wb') as f:
        pickle.dump(_l_dev_text, f, pickle.HIGHEST_PROTOCOL)
    print("Done writing development set from pickle.")

print("development set")
print("\t LAN\t size \t avg length")
for l in _l_dev_text.keys():
    print("\t", l, "\t", len(_l_dev_text[l]), "\t", sum([len(x.split()) for x in _l_dev_text[l]])/len(_l_dev_text[l]))

# And then the training set. This takes bit more time...
try:
    with open('data/train.txt.pickle', 'rb') as f:
        _l_trn_text = pickle.load(f)
        print("Done reading training set from pickle.")
except IOError:
    _l_trn_text = read_data('data/train.txt')
    print("Done tokenising training set.")
    with open('data/train.txt.pickle', 'wb') as f:
        pickle.dump(_l_trn_text, f, pickle.HIGHEST_PROTOCOL)
    print("Done writing training set from pickle.")

print("training set")
print("\t LAN\t size \t avg length")
for l in _l_trn_text.keys():
    print("\t", l, "\t", len(_l_trn_text[l]), "\t", sum([len(x.split()) for x in _l_trn_text[l]])/len(_l_trn_text[l]))
    


# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from matplotlib.legend_handler import HandlerLine2D
import xgboost as xgb

# In[6]:


_X_training = []
_y_training = []
for l in _l_trn_text.keys():
    for s in _l_trn_text[l]:
        _X_training.append(s)
        _y_training.append(l)
_X_training = np.array(_X_training, dtype=object)
_y_training = labelEncoder.transform(_y_training)


_X_dev = []
_y_dev = []
for l in _l_dev_text.keys():
    for s in _l_dev_text[l]:
        _X_dev.append(s)
        _y_dev.append(l)
_X_dev = np.array(_X_dev, dtype=object)
_y_dev = labelEncoder.transform(_y_dev)



# In[7]:


if not os.path.exists('data/' + 'dev' + '.POS.txt') or not os.path.exists('data/' + 'train' + '.POS.txt'):
    import frog

    frog = frog.Frog(frog.FrogOptions(parser=False))

    for t in ['dev', 'train']:
        with open('data/' + t + '.POS.txt', 'w') as out:
            with open('data/' + t + '.txt', 'r') as f:
                for line in f:
                    sentence, tag = line.strip().split("\t")
                    froggo = frog.process(sentence)
                    postext = []
                    for w in froggo:
                        postext.append(w['pos'].split("(")[0])
                    out.write(" ".join(postext) + "\t" + tag + "\n")


# In[8]:


_X_pos_training = []
_y_pos_training = []
with open('data/train.POS.txt', 'r') as f:
    for line in f:
        sentence, tag = line.strip().split("\t")
        _X_pos_training.append(sentence)
        _y_pos_training.append(tag)
_X_pos_training = np.array(_X_pos_training)
_y_pos_training = labelEncoder.transform(_y_pos_training)

_X_pos_dev = []
_y_pos_dev = []
with open('data/dev.POS.txt', 'r') as f:
    for line in f:
        sentence, tag = line.strip().split("\t")
        _X_pos_dev.append(sentence)
        _y_pos_dev.append(tag)
_X_pos_dev = np.array(_X_pos_dev)
_y_pos_dev = labelEncoder.transform(_y_pos_dev)


# In[9]:


import six
from abc import ABCMeta
import numpy as np
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, C=1.0, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = [] # fuggly

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # so we don't have to cast X to floating point
        Y = Y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)

        # flugglyness
        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)
        
        return self.classes_[np.argmax(D, axis=0)]
        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)


# # 1 Determine cv score on training material

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import fasttext
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner

from prefit_voting_classifier import PrefitVotingClassifier


# ## Models
# 
# ### mnb_st (top)

# In[11]:


mnb_st_pl = Pipeline([('union', FeatureUnion([('char', TfidfVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', None),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', MultinomialNB(alpha=0.05))])


# ### sgd_st (top)

# In[12]:


sgd_st_pl = Pipeline([('union', FeatureUnion([('char', TfidfVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', SGDClassifier(alpha=0.0001,max_iter=4,shuffle=True))])


# ### fat1 (top)

# In[13]:


import FastTextClassifier


# In[14]:


fat_pl = FastTextClassifier.FastTextClassifier('fasttext.train.txt', 'model', 
                                        minc=1, 
                                        ngram=5, 
                                        epoch=150,
                                        minn=3, 
                                        maxn=7, 
                                        thread=2, 
                                        dim=175,
                                        ws=5,
                                        nbucket= 2000000)


# ### mkn_lm (top)

# In[17]:


import MKNClassifier

mkn_pl = MKNClassifier.MKNClassifier(n=3)


# ### mnb_sc (mid)

# In[ ]:


mnb_sc_pl = Pipeline([('union', FeatureUnion([('char', CountVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', CountVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', None),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', MultinomialNB(alpha=0.05))])


# ### sgd_sc (mid)

# In[ ]:


sgd_sc_pl = Pipeline([('union', FeatureUnion([('char', CountVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', CountVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', SGDClassifier(alpha=0.1,max_iter=4,shuffle=True))])


# ### nbs_st (mid)

# In[ ]:


nbs_st_pl = Pipeline([('union', FeatureUnion([('char', TfidfVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', None),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', NBSVM(C=0.1, alpha=1e-05))])


# ### knn_st (mid)

# In[ ]:


knn_st_pl = Pipeline([('union', FeatureUnion([('char', TfidfVectorizer(analyzer='char', ngram_range=(1,8))),
                                              ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b"))])),
                      ('scaler', None),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', weights='uniform'))])


# ### mnb_ps (mid)

# In[ ]:


mnb_ps_pl = Pipeline([('word', CountVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b")),
                      ('scaler', None),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', MultinomialNB(alpha=0.0001))])


# ### sgd_ps (mid)

# In[ ]:


sgd_ps_pl = Pipeline([('word', CountVectorizer(analyzer='word', ngram_range=(1,6),token_pattern=u"(?u)\\b\\w+\\b")),
                      ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)),
                      ('kbest', SelectKBest(k='all')), 
                      ('clf', SGDClassifier(alpha=1e-06,max_iter=4,shuffle=True))])


# ## Run the models

# In[ ]:


#[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not x.startswith("mnb")]


# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True)

# subset = np.random.randint(len(_X_training), size=1000)
# _X_training = _X_training[subset]
# _y_training = _y_training[subset]


# In[ ]:


# y_predictions: [[weight1, predictions1], ..., [weightn, predictionsn]]
def ensemble(y_true, y_predictions):
    for _, preds in y_predictions:
        if len(preds) != len(y_true):
            print("WTF")
    
    scores = []
    for i in range(len(y_true)):
        score = 0
        for weight, preds in y_predictions:
            score += weight * (-1.0, 1.0)[preds[i]]

        if score == 0.0:
            score = random.randint(0,1)
        scores.append((0, 1)[score > 0])
    return np.array(scores)


# In[ ]:


import random

# weighted blends
# [('name', [prediction1, ..., predictionN]), ..., ('name', [prediction1, ..., predictionN])]
def frw(models, y_true):   
    best_acc = 0
    best_norm_weights = []
   
    x_range = int(min(100000, 10**len(models))/10)+1
    for x in range(x_range):
        asd = []
        for name, predictions in models:
            asd.append([random.randint(-100,100), predictions])
    
    #print(asd)
    
        acc = accuracy_score(y_true, ensemble(y_true, asd))

        if acc > best_acc:
            best_acc = acc
            weights = [row[0] for row in asd]
            #best_norm_weights = [float(i)/sum(weights) for i in weights]
            print(x, acc, weights)
    #print(acc)
    return (best_acc, best_norm_weights)


# In[18]:

# 
def get_data(name, X1, X2, y1, y2):
    if "_ps_" in name:
        return (X2, y2)
    else:
        return (X1, y1)

def make_pipelines():

    all_pipelines = {
                     'mnb_st_pl': {'pipeline': mnb_st_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
                     'mkn_pl': {'pipeline': mkn_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
                     'sgd_st_pl': {'pipeline': sgd_st_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
                     'fat_pl': {'pipeline': fat_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
                     'mnb_sc_pl': {'pipeline': mnb_sc_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
                     'sgd_sc_pl': {'pipeline': sgd_sc_pl,
                                   'scores': np.array([]),
                                   'predictions': []},
    ##                  'nbs_st_pl': {'pipeline': nbs_st_pl,
    ##                                'scores': [],
    ##                                'predictions': []},
    #                 'knn_st_pl': {'pipeline': knn_st_pl,
    #                               'scores': np.array([]),
    #                               'predictions': []},
    #                 'mnb_ps_pl': {'pipeline': mnb_ps_pl,
    #                               'scores': np.array([]),
    #                               'predictions': []},
    #                 'sgd_ps_pl': {'pipeline': sgd_ps_pl,
    #                               'scores': np.array([]),
    #                               'predictions': []},
                    }
    
    combined_pipelines = {'combinedsoft': {'scores': np.array([])},
                          'combinedhard': {'scores': np.array([])},
                          'blended': {'scores': np.array([])},
                          'stackxgbsoft': {'scores': np.array([])},
                          'stackxgbhard': {'scores': np.array([])},
                          'ensemble': {'scores': np.array([])},
                         }
    return (all_pipelines, combined_pipelines)

print("================================================")
print("=========== PART 1 =============================")

import operator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

(all_pipelines, combined_pipelines) = make_pipelines()
for train_index, test_index in kf.split(_X_training):
    X_train, X_test = _X_training[train_index], _X_training[test_index]
    y_train, y_test = _y_training[train_index], _y_training[test_index]
    
    X_pos_train, X_pos_test = _X_pos_training[train_index], _X_pos_training[test_index]
    y_pos_train, y_pos_test = _y_pos_training[train_index], _y_pos_training[test_index]
    
    local_predictions = []
    local_predictions_proba = []
    
    #############################
    print("== individual models")
    for pipeline_name in all_pipelines.keys():
        pipeline = all_pipelines[pipeline_name]['pipeline']
        #pipeline.fit(X_train, y_train)
        pipeline.fit(*get_data(pipeline_name, X_train, X_pos_train, y_train, y_pos_train))
        predictions = pipeline.predict(X_test)
        #score = pipeline.score(X_test, y_test)
        score = pipeline.score(*get_data(pipeline_name, X_test, X_pos_test, y_test, y_pos_test))

        local_predictions.append((pipeline_name, list(predictions)))
        if hasattr(pipeline, 'predict_proba'):
            #predictions_proba = pipeline.predict_proba(X_test)
            predictions_proba = pipeline.predict_proba(get_data(pipeline_name, X_test, X_pos_test, None, None)[0])
            local_predictions_proba.append((pipeline_name, list(predictions_proba)))

        all_pipelines[pipeline_name]['predictions'] += [list(predictions)]
        
        all_pipelines[pipeline_name]['scores'] = np.concatenate((all_pipelines[pipeline_name]['scores'], [score]))
        
        print(pipeline_name, score)
    
    ################################
    print("== combined soft models")
    #eclf1 = VotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not x.startswith("sgd") and not "_ps_" in x],
    eclf1 = PrefitVotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not x.startswith("sgd") and not "_ps_" in x],
                             voting='soft')
    #eclf1 = eclf1.fit(X_train, y_train)
    eclf1_score = eclf1.score(X_test, y_test)
    combined_pipelines['combinedsoft']['scores'] = np.concatenate((combined_pipelines['combinedsoft']['scores'], [eclf1_score]))
    print(eclf1_score)
    
    ################################
    print("== combined hard models")
    #eclf2 = VotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not "_ps_" in x],
    eclf2 = PrefitVotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not "_ps_" in x],
                             voting='hard')
    #eclf2 = eclf2.fit(X_train, y_train)
    eclf2_score = eclf2.score(X_test, y_test)
    combined_pipelines['combinedhard']['scores'] = np.concatenate((combined_pipelines['combinedhard']['scores'], [eclf2_score]))
    print(eclf2_score)
    
    ##########################
    print("== blended models")
    (frw_acc, frw_weights) = frw(local_predictions ,y_test)
    combined_pipelines['blended']['scores'] = np.concatenate((combined_pipelines['blended']['scores'], [frw_acc]))

#    ################################
#    print("== stacked xgboost hard")
#    X_xgbh = np.rot90(np.array([gg[1] for gg in local_predictions]))
#    xgbh = xgb.XGBClassifier()
#    xgbh.fit(X_xgbh, y_train)
#    xgbh_score = xgbh.score(X_xgbh, y_test)
#    combined_pipelines['stackxgbhard']['scores'] = np.concatenate((combined_pipelines['stackxgbhard']['scores'], [xgbh_score]))
#    
#    ###############################
#    print("== stacked xgboost soft")
#    X_xgbs = np.rot90(np.array([gg[1] for gg in local_predictions_proba]))
#    xgbs = xgb.XGBClassifier()
#    xgbs.fit(X_xgbs, y_train)
#    xgbs_score = xgbs.score(X_xgbs, y_test)
#    combined_pipelines['stackxgbsoft']['scores'] = np.concatenate((combined_pipelines['stackxgbsoft']['scores'], [xgbs_score]))

#     print("\tWorking with:")

    brew_ensemble_l1 = Ensemble([mnb_st_pl, fat_pl, sgd_sc_pl])
    brew_ensemble_l2 = Ensemble([sgd_st_pl, mkn_lm])#, knn_st])
    stack = EnsembleStack(cv=3)
    stack.add_layer(brew_ensemble_l1)
    stack.add_layer(brew_ensemble_l2)
    sclf = EnsembleStackClassifier(stack)
    #sclf.fit(X_train, y_train)
    sclf_score = sclf.score(X_test, y_test)
    combined_pipelines['ensemble']['scores'] = np.concatenate((combined_pipelines['ensemble']['scores'], [sclf_score]))
    print("sclf_score", sclf_score)

print()
print("= SUMMARY")
best_predictions = []
for pipeline_name in all_pipelines.keys():
    index, value = max(enumerate(all_pipelines[pipeline_name]['scores']), key=operator.itemgetter(1))
    print(pipeline_name, index, value)
    best_predictions.append((pipeline_name, all_pipelines[pipeline_name]['predictions'][index]))
        
for pipeline_name in all_pipelines.keys():
    print(pipeline_name)
    print("\tMin:", all_pipelines[pipeline_name]['scores'].min())
    print("\tMax:", all_pipelines[pipeline_name]['scores'].max())
    print("\tAvg:", all_pipelines[pipeline_name]['scores'].mean())
    print("\tStd:", all_pipelines[pipeline_name]['scores'].std())
  
for pipeline_name in combined_pipelines.keys():
    print(pipeline_name)
    print("\tMin:", combined_pipelines[pipeline_name]['scores'].min())
    print("\tMax:", combined_pipelines[pipeline_name]['scores'].max())
    print("\tAvg:", combined_pipelines[pipeline_name]['scores'].mean())
    print("\tStd:", combined_pipelines[pipeline_name]['scores'].std())
print()    


# # 2 Determine score on development material

# In[ ]:

print("================================================")
print("=========== PART 2 =============================")

(all_pipelines, combined_pipelines) = make_pipelines()

################################
print("== combined soft models")
eclf3 = VotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys() if not x.startswith("sgd")],
                         voting='soft')
eclf3 = eclf3.fit(_X_training, _y_training)
eclf3_score = eclf3.score(_X_dev, _y_dev)
combined_pipelines['combinedhard']['scores'] = np.concatenate((combined_pipelines['combinedhard']['scores'], [eclf3_score]))
print(eclf3_score)

local_predictions2 = []
#############################
print("== individual models")
for pipeline_name in all_pipelines.keys():
    pipeline = all_pipelines[pipeline_name]['pipeline']
    pipeline.fit(_X_training, _y_training)
    predictions = pipeline.predict(_X_dev)
    score = pipeline.score(_X_dev, _y_dev)

    local_predictions2.append((pipeline_name, list(predictions)))

    all_pipelines[pipeline_name]['predictions'] += [list(predictions)]

    all_pipelines[pipeline_name]['scores'] = np.concatenate((all_pipelines[pipeline_name]['scores'], [score]))

    print(pipeline_name, score)

################################
print("== combined hard models")
eclf4 = VotingClassifier(estimators=[(x, all_pipelines[x]['pipeline']) for x in all_pipelines.keys()],
                         voting='hard')
eclf4 = eclf4.fit(_X_training, _y_training)
eclf4_score = eclf4.score(_X_dev, _y_dev)
combined_pipelines['combinedhard']['scores'] = np.concatenate((combined_pipelines['combinedhard']['scores'], [eclf4_score]))
print(eclf4_score)

##########################
print("== blended models")

(frw_acc, frw_weights) = frw(local_predictions2 , _y_dev)
combined_pipelines['blended']['scores'] = np.concatenate((combined_pipelines['blended']['scores'], [frw_acc]))

print()
print("= SUMMARY")

for pipeline_name in all_pipelines.keys():
    print(pipeline_name)
    print("\tMin:", all_pipelines[pipeline_name]['scores'].min())
    print("\tMax:", all_pipelines[pipeline_name]['scores'].max())
    print("\tAvg:", all_pipelines[pipeline_name]['scores'].mean())
    print("\tStd:", all_pipelines[pipeline_name]['scores'].std())
  
for pipeline_name in combined_pipelines.keys():
    print(pipeline_name)
    print("\tMin:", combined_pipelines[pipeline_name]['scores'].min())
    print("\tMax:", combined_pipelines[pipeline_name]['scores'].max())
    print("\tAvg:", combined_pipelines[pipeline_name]['scores'].mean())
    print("\tStd:", combined_pipelines[pipeline_name]['scores'].std())
# # 3 Determine cv score on training and development material

# # 4 Create submission

# In[ ]:


# In[ ]:


