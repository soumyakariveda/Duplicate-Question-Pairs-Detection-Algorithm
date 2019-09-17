import pandas as pd
import numpy as np
import collections
# from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import sys
import string
from textblob import Word
# from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from fuzzywuzzy import fuzz
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

def preprocessing(d):
        df=d[0]
        df = df.lower()
        #df.translate(None,string.punctuation)
        exclude = set(string.punctuation)
        df = ''.join(ch for ch in df if ch not in exclude)
        list_df=df.split()
        lemmatize_list=[]
        for i in list_df:
            w=Word(i)
            lemmatize_list.append(w.lemmatize())
#         stop_words=set(stopwords.words('english'))
#         filtered_list = [w for w in lemmatize_list if not w in stop_words]
#         filtered_sentence =' '.join(filtered_list)
        filtered_sentence =' '.join(lemmatize_list)
        return filtered_sentence

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def tfidf_word_match_share(row):
    stops = set(stopwords.words("english"))
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    eps = 5000
    words = (" ".join(row)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    try:
        return np.sum(shared_weights) / np.sum(total_weights)
    except ZeroDivisionError:
        return 0

def word_match_share(row):
    stops = set(stopwords.words("english"))
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    try:
        return (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    except ZeroDivisionError:
        return 0

def tf_idf_ngram(d):
    l = [d[0],d[1]]
    vectorizer = TfidfVectorizer(ngram_range = (1,4))
    vectorizer.fit(l)
    question1 = vectorizer.transform([l[0]])
    question2 = vectorizer.transform([l[1]])
    arr1 = question1.toarray()[0]
    arr2 = question2.toarray()[0]
    arr = np.array([arr1,arr2])
    return arr

def cosine_dist(df):
    mat=df
    a = mat[0]
    b = mat[1]
    return np.dot(a,b)

def diff_length(d):
    q1 = d[0]
    q2 = d[1]
    w1 = q1.split()
    w2 = q2.split()
    return abs(len(w1)-len(w2))

def jaccard_similarity(d):
    q1 = d[0]
    q2 = d[1]
    w1 = q1.split()
    w2 = q2.split()
    s1 = set(w1)
    s2 = set(w2)
    return float(len(s1 & s2))/len(s1 | s2)

def fuzz_features(df):
    q1 = df[0]
    q2 = df[1]
    fuzz_ratio = fuzz.ratio(q1,q2)
    return fuzz_ratio

def bray_curtis_dist(df):
    mat = df
    a = mat[0]
    b = mat[1]
    numerator = 0
    denominator = 0
    n = a.size
    for i in range (0,n):
        numerator += abs(a[i]-b[i])
    for i in range (0,n):
        denominator += abs(a[i]+b[i])
    try:
        return float(numerator)/denominator
    except ZeroDivisionError:
        return 1

def euclidian_dist(df):
    mat = df
    a = mat[0]
    b = mat[1]
    return np.linalg.norm(a-b)

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0

def question_freq(train_test_df):
    df1 = train_test_df[['question1']].copy()
    df2 = train_test_df[['question2']].copy()
    df1_test = train_test_df[['test_ques1']].copy()
    df2_test = train_test_df[['test_ques2']].copy()
    df2.rename(columns = {'question2':'question1'},inplace=True)
    df2_test.rename(columns = {'test_ques2':'question1'},inplace=True)
    df1_test.rename(columns = {'test_ques1':'question1'},inplace=True)
    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    train_questions.drop_duplicates(subset = ['question1'],inplace=True)
    train_questions.reset_index(inplace=True,drop=True)
    questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
    train_cp = pd.DataFrame(train_test_df[['question1','question2','id','is_duplicate']])
    test_cp = pd.DataFrame(train_test_df[['test_ques1','test_ques2','test_id']])
    test_cp.rename(columns={'test_ques1':'question1'},inplace=True)
    test_cp.rename(columns={'test_ques2':'question2'},inplace=True)
    # train_cp.drop(['qid1','qid2'],axis=1,inplace=True)
    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id':'id'},inplace=True)
    comb = pd.concat([train_cp,test_cp],sort=True)
    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)
    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()
    comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    train_q1_freq = comb[pd.to_numeric(comb['is_duplicate']) >= 0][['q1_freq']]
    train_q2_freq = comb[pd.to_numeric(comb['is_duplicate']) >= 0][['q2_freq']]
    test_q1_freq = comb[pd.to_numeric(comb['is_duplicate']) < 0][['q1_freq']]
    test_q2_freq = comb[pd.to_numeric(comb['is_duplicate'])< 0][['q2_freq']]
    train_test_df_freq = pd.DataFrame(train_q1_freq)
    train_test_df_freq['q2_freq'] = train_q2_freq[['q2_freq']].copy()
    train_test_df_freq['test_q1_freq'] = test_q1_freq[['q1_freq']].copy()
    train_test_df_freq['test_q2_freq'] = test_q2_freq[['q2_freq']].copy()
    return train_test_df_freq

def getQuestions(df):
    data_frame = df[['question1','question2']].copy()
    return data_frame

class QuestionPairs(object):

    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile
        self.__xgb = xgb.XGBClassifier()
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.predicted_labels = None
        self.train_features = pd.DataFrame()
        self.test_features = pd.DataFrame()

    def trainingData(self):
        train = pd.read_csv(self.trainFile,low_memory = False)
        train["question1"].fillna("no", inplace = True)
        train["question2"].fillna("no", inplace = True)
        Total_Data_Frame = pd.DataFrame(train)
        Total_Data_Frame = Total_Data_Frame[['id','qid1','qid2','question1','question2','is_duplicate']].copy()
        Total_Data_Frame = Total_Data_Frame[(Total_Data_Frame.is_duplicate == '0') | (Total_Data_Frame.is_duplicate == '1')]
        Total_Data_Frame = Total_Data_Frame[:]
        self.train_labels = Total_Data_Frame['is_duplicate']
        self.train_data = Total_Data_Frame

    def testingData(self):
        test = pd.read_csv(self.testFile,low_memory = False)
        test["question1"].fillna("no", inplace = True)
        test["question2"].fillna("no", inplace = True)
        Test_Total_Data_Frame = pd.DataFrame(test)
        Test_Total_Data_Frame = Test_Total_Data_Frame[['id','question1','question2']].copy()
        self.test_data = Test_Total_Data_Frame

    # def getQuestions(df):
    #     data_frame = df[['question1','question2']].copy()
    #     return data_frame

    def data(self):
        self.trainingData()
        self.testingData()

    def trainFeaturesApply(self):
        data_frame = (self.train_data).apply(getQuestions, axis=1)
        data_frame_ngram = data_frame.apply(tf_idf_ngram, axis=1)
        test_data_frame = (self.test_data).apply(getQuestions, axis=1)
        train_test_df = pd.DataFrame(self.train_data)
        train_test_df['test_ques1'] = self.test_data[['question1']].copy()
        train_test_df['test_ques2'] = self.test_data[['question2']].copy()
        train_test_df['test_id'] = self.test_data[['id']].copy()
        # frequency = train_test_df.apply(question_freq, axis=1)
        frequency = question_freq(train_test_df)
        self.train_features['tfidf_match_score'] = data_frame.apply(tfidf_word_match_share,axis = 1)
        self.train_features['match_score'] = data_frame.apply(word_match_share,axis = 1)
        self.train_features['q1_freq'] = frequency[['q1_freq']].copy()
        self.train_features['q2_freq'] = frequency[['q2_freq']].copy()
        self.train_features['cosine_dist_ngram'] = data_frame_ngram.apply(cosine_dist)
        self.train_features['diff_length'] = data_frame.apply(diff_length, axis=1)
        self.train_features['jaccard'] = data_frame.apply(jaccard_similarity, axis=1)
        self.train_features['fuzz_ratio'] = data_frame.apply(fuzz_features, axis=1)
        self.train_features['bray_curtis_dist'] = data_frame_ngram.apply(bray_curtis_dist)
        self.train_features['euclidian_dist_ngram'] = data_frame_ngram.apply(euclidian_dist)
        self.train_features['cosine_dist_ngram'].fillna(self.train_features['cosine_dist_ngram'],inplace=True)
        self.train_features['diff_length'].fillna(self.train_features['diff_length'],inplace=True)
        self.train_features['jaccard'].fillna(self.train_features['jaccard'],inplace=True)
        self.train_features['fuzz_ratio'].fillna(self.train_features['fuzz_ratio'],inplace=True)
        self.train_features['bray_curtis_dist'].fillna(self.train_features['bray_curtis_dist'],inplace=True)
        self.train_features['euclidian_dist_ngram'].fillna(self.train_features['euclidian_dist_ngram'],inplace=True)
        print list(self.train_features)

    def testFeaturesApply(self):
        data_frame = (self.test_data).apply(getQuestions, axis=1)
        data_frame_ngram = data_frame.apply(tf_idf_ngram, axis=1)
        train_data_frame = (self.train_data).apply(getQuestions, axis=1)
        train_test_df = pd.DataFrame(self.train_data)
        train_test_df['test_ques1'] = self.test_data[['question1']].copy()
        train_test_df['test_ques2'] = self.test_data[['question2']].copy()
        train_test_df['test_id'] = self.test_data[['id']].copy()
        # frequency = train_test_df.apply(question_freq, axis=1)
        frequency = question_freq(train_test_df)
        self.test_features['tfidf_match_score'] = data_frame.apply(tfidf_word_match_share,axis = 1)
        self.test_features['match_score'] = data_frame.apply(word_match_share,axis = 1)
        self.test_features['q1_freq'] = frequency[['test_q1_freq']].copy()
        self.test_features['q2_freq'] = frequency[['test_q2_freq']].copy()
        self.test_features['cosine_dist_ngram'] = data_frame_ngram.apply(cosine_dist)
        self.test_features['diff_length'] = data_frame.apply(diff_length, axis=1)
        self.test_features['jaccard'] = data_frame.apply(jaccard_similarity, axis=1)
        self.test_features['fuzz_ratio'] = data_frame.apply(fuzz_features, axis=1)
        self.test_features['bray_curtis_dist'] = data_frame_ngram.apply(bray_curtis_dist)
        self.test_features['euclidian_dist_ngram'] = data_frame_ngram.apply(euclidian_dist)
        self.test_features['cosine_dist_ngram'].fillna(self.test_features['cosine_dist_ngram'],inplace=True)
        self.test_features['diff_length'].fillna(self.test_features['diff_length'],inplace=True)
        self.test_features['jaccard'].fillna(self.test_features['jaccard'],inplace=True)
        self.test_features['fuzz_ratio'].fillna(self.test_features['fuzz_ratio'],inplace=True)
        self.test_features['bray_curtis_dist'].fillna(self.test_features['bray_curtis_dist'],inplace=True)
        self.test_features['euclidian_dist_ngram'].fillna(self.test_features['euclidian_dist_ngram'],inplace=True)
        print list(self.test_features)

    def trainxgboost(self):
        self.__xgb = xgb.XGBClassifier(learning_rate = 0.7,random_state = 1)
        self.__xgb.fit(self.train_features,self.train_labels)
        filename = 'Error404BrainNotFound_model.pkl'
        pickle.dump(self.__xgb, open(filename, 'wb'))
        xgb.plot_importance(self.__xgb)
        plt.show()
        # return model

    def testxgboost(self):
        pred_values_df = self.test_data[['id']].copy()
        filename = 'Error404BrainNotFound_model.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        pred_values = loaded_model.predict_proba(self.test_features)
        pred_values_df['is_duplicate'] = pred_values[:,1]
        pred_values_df.to_csv("s.csv",index = False)

if __name__ == "__main__":
    train_data_name = sys.argv[1]
    test_data_name = sys.argv[2]
    model = QuestionPairs(train_data_name,test_data_name)
    model.data()
    model.trainFeaturesApply()
    model.testFeaturesApply()
    model.trainxgboost()
    model.testxgboost()
