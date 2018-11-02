import numpy as np
import pandas as pd
import re
import tensorflow as tf
import random


####################################################
# cut words function                               #
####################################################
#문장의 내용을 2개씩 커팅하는 것이다.
def cut(contents):
    results = []
    for content in contents:#리스트에 있는 모든 문서 단어들 
        words = content.split(' ')# ' '기준으로 스플릿
        result = []# 결과는 
        for word in words:# word 개수만큼 
            result.append(word)# word를 result에 넣고 
        results.append(' '.join([token for token in result]))
    return results

####################################################
# divide train/test set function                   #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
def make_input(documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# make output function                             #
####################################################
#make output은 그야말로 output을 나타내는 것

def make_output(points):# cls는 clssfy
    results = np.zeros((len(points),3))
    for idx, point in enumerate(points):
        if (point == 'Criminal'):# 
            results[idx,0] = 1
        elif(point =='Civil'):
            results[idx,1] = 1
        else:
            results[idx,2] =1
    return results#각 문서 종류에맞춰 아웃풋을 형사, 민사 행정 순으로 1,0,0 / 0,1,0 / 0,0,1 로 임베딩해줌

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length
# document_lenth를 비교. #나는 10밖에안줬기 때문에 무조껀 10나옴
####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path, eng=True, num=True, punc=False):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents = []#내용
    cls = []# 문서종류
    for idx,doc in enumerate(corpus):
        if isNumber(doc[0]) is False:
            contents.append(doc[0])
            cls.append(doc[1])
        if idx % 100000 is 0:
            print('%d docs / %d save' % (idx, len(contents)))
    return contents, cls

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False