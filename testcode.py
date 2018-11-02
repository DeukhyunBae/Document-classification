
# coding: utf-8

# In[7]:


import os
from konlpy.tag import Twitter
from konlpy.utils import pprint
import collections
import time
import os
import xlrd
import xlwt
import re
import numpy as np
import codecs
import gensim
print("파일 경로와 파일이름을 입력하시오.")
var=input()

file= open(var,'r',-1,'UTF8')
## e = 행정 , y= 형사 , m= 민사
for i in range(300):
    text=file.readline()
    if(text=="【이 유】\n"):
        break;
f=file.read()

# In[8]:


def get_list(text):
    spliter = Twitter()
    split= spliter.pos(text, norm=True, stem= True)
    return_list =[]
    for n, c in split:
        j=0 #중복단어 트리거
        i=0 # 루프 트리거
        if( c =='Verb'):
            i=0 # 루프 트리거
            for i in range(len(return_list)):
                if(n== return_list[i]):
                    j=1# 만약 중복되면 j=1
        elif( c == 'Noun'):
            i=0 # 루프 트리거
            for i in range(len(return_list)):
                if(n== return_list[i]):
                    j=1
        else:
            j=1
        if(j==0):#중복된 단어가 없을경우에만
            return_list.append(n)# 리스트에 단어를 삽입
            #output file에다 저
    return return_list


# In[9]:


text=get_list(f)
file.close()
print("##############################")
print("#동사와 명사를 추출한 결과 값#")
print(text)
print("##############################")
#if(len(text)<10):
#   print("Error occur because of length of size of words in documents")
#    break;

# In[10]:

model=gensim.models.KeyedVectors.load('./union_model_Noverlap2')#모델을 읽고
print("#####################################")
print("#형사와 가장 가까운 단어 테이블 선정#")
temp=model.most_similar("형사",topn=100)
word1=[]
for i in range(len(temp)):
    word1.append(temp[i][0])
print("#민사와 가장 가까운 단어 테이블 선정#")
word2=[]
temp=model.most_similar("민사",topn=100)
i=0
for i in range(len(temp)):
    word2.append(temp[i][0])
print("#행정과 가장 가까운 단어 테이블 선정#")
print("#####################################")
word3=[]
temp=model.most_similar("행정",topn=100)
i=0
for i in range(len(temp)):
    word3.append(temp[i][0])
accel=1.05
accel2=0.005
valuelist1=[]
valuelist2=[]
valuelist3=[]
for i in range(len(text)):
    top=0
    for j in range(100):
        accelator=accel-(accel2*j)
        temp=(model.similarity(text[i],word1[j]))*accelator
        if(top< temp):
            top=temp
    valuelist1.append(top)#형사모델 가장 큰값 순서대로
for i in range(len(text)):
    top=0
    for j in range(100):
        accelator=accel-(accel2*j)
        temp=(model.similarity(text[i],word2[j]))*accelator
        if(top< temp):
            top=temp
    valuelist2.append(top)# 민사모델 가장 큰값 순서대로
for i in range(len(text)):
    top=0
    for j in range(100):
        accelator=accel-(accel2*j)
        temp=(model.similarity(text[i],word3[j]))*accelator
        if(top< temp):
            top=temp
    valuelist3.append(top)# 행정모델 큰값 순서대로 
print("#########################################################################")
print("######################형사와 가장 가까운 테이블 :########################")
print(word1)
print("#형사와 가장 가까운 단어와 코사인 유사도 거리 스칼라값과 가중치를 곱한값#")
print(valuelist1)
print("#########################################################################")
print("#########################################################################")
print("######################민사와 가장 가까운 테이블 :########################")
print(word2)
print("#민사와 가장 가까운 단어와 코사인 유사도 거리 스칼라값과 가중치를 곱한값#")
print(valuelist2)
print("#########################################################################")
print("#########################################################################")
print("######################행정과 가장 가까운 테이블 :########################")
print(word3)
print("#행정과 가장 가까운 단어와 코사인 유사도 거리 스칼라값과 가중치를 곱한값#")
print(valuelist3)
print("#########################################################################")


# In[12]:


result1=0.0
result2=0.0
result3=0.0
for i in range(len(valuelist1)):
    result1=result1+valuelist1[i]
for i in range(len(valuelist2)):
    result2=result2+valuelist2[i]
for i in range(len(valuelist3)):
    result3=result3+valuelist3[i]
if(result1> result2):
    valuelist=valuelist1
    results=result1
else:
    valuelist=valuelist2
    results=result2
if(results<result3):
    valuelist=valuelist3
print("##################################################")
print("#형사 , 민사 , 행정과 가장 가까운 사건과 매칭중..#")
print("##################################################")


# In[13]:


TW=[0]*10 #top word
TS=[0]*10 #top similar
for j in range(len(valuelist)):#한 문장 단어개수 10개 추리기
        if( valuelist[j]<1):# 너무 타겟 되는게 아니라 
            for k in range(10):#소팅시키는작업 
                if(TS[k]<valuelist[1]):#만약 더 큰숫자가 나타난다면
                    for l in range(10 - k):# 맨뒤부터 차례대로 앞숫자 댕겨와서
                        if(k<9):
                            TW[9-l]= TW[8-l]
                            TS[9-l]= TS[8-l]#순서대로 워드 밀어주는 방법
                    TW[k]=text[j]
                    TS[k]=valuelist[j]
                    break;#한번 걸리면 바로 브레이크걸어준다 


# In[25]:

print("####################################################")
print("#코사인 유사도 거리값이 가장 가까운 단어 10개 추출!#")
print("####################################################")
print(TW)
out= open('./sample.txt','w',-1,encoding = 'UTF8')


# In[27]:


for i in range(10):
    out.write(TW[i])
    if(TW[i]!=9):
        out.write(' ')
    
out.close()

# In[28]:


import os
import time
import datetime
from tensorflow import flags
import tensorflow as tf
import numpy as np
import cnn_tool as tool
import argparse
import codecs

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
#        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
#            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
#            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.scores, logits= self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels= self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

data_path='./data.csv'
contents, cls = tool.loading_rdata(data_path, eng=True, num=True, punc=False)
contents = tool.cut(contents)
max_document_length = 10
x, vocabulary, vocab_size = tool.make_input(contents,max_document_length)
print('사전단어수 : %s' % (vocab_size))
y = tool.make_output(cls)
x_train, x_test, y_train, y_test = tool.divide(x,y,train_prop=1)
# tranform document to vector
# train 데이터와 dev데이터를 나눠주는 역활을한다. 
# 다른코드는 파일로 저장을하는데 여기서는 그냥 함수콜을 해서 리턴해서 로컬변수로 저장하는듯함

# 데이터 타입과 value들을 말해주는데 fllags의 모든 항목들을 보여준다.
# 3. train the model and test
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=vocab_size,
                      embedding_size=100,
                      filter_sizes=[3,4,5],
                      num_filters=100,
                      l2_reg_lambda=0.0)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

 
        # Initialize all variables


# 데이터 타입과 value들을 말해주는데 fllags의 모든 항목들을 보여준다.
# 3. train the model and test

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("checkpoint 폴더경로를 입력하시오 (예: ./runs/숫자/checkpoints/)")
        var=input()
        ckpt = tf.train.latest_checkpoint(var)
        saver.restore(sess, ckpt)
        print("Model loading is ok")
print("#############################")        
print("#학습된 모델 불러들이는중...#")
print("#############################")


# In[86]:


import pandas as pd
import cnn_tool as tool
f = open("./vocab.txt", "r", encoding='utf8')
data= f.read()
line = []
line=data.split("\n")
#print(line)

#data2 = file.read()
#print(data2)
#line2=[]
#line2=data2.split('\n')
def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False
def find(line, data):
    check = 1
    for i in range(len(line)):
        if(line[i]==data):
            check=0
            return 0
    return 1
#corpus = pd.read_table(data_file_path, encoding="utf-8")
#corpus = np.array(corpus)

data_file_path  = './sample.txt'
file = open(data_file_path, 'r', encoding='UTF8')
contents=file.readline()
content=[]
content=contents.split()
#for idx,doc in enumerate(corpus):
#    if isNumber(doc) is False:
#        contents.append(doc)
    #if idx % 100000 is 0:
    #    print('%d docs / %d save' % (idx, len(contents)))        
######################################
#x =[]
#print(content)
#content=[]
#contents = []#내용
##for i in range(len(contents)):
#    temp=contents[i].split()
#    x1=[]
#    for j in range(len(temp)):
#        x1.append(line.index(temp[j]))
#    content.append(temp)
#    x.append(x1)#
#x= np.array(x)
#print(x)
##########################################    
#for i in range(0,len(line2)):
#    data3 =[]
#    data3=line2[i].split(',')# 던어랑 클래스 분리
data=[]
x_array=[]
for j in range(10):
    if(find(line,content[j])==1):
        num=0
        x_array.append(num)
    else:
        num=line.index(content[j])
        x_array.append(num)
 #   x_array.append(line.index(content[j]))#숫자로 
np_X=np.array(x_array)
x_data=np.array([np_X])
    #X임베딩 완료        X값 을 임베딩하는건 포문 돌려야함
#Y부분( 클래스부분)은 한번에 처리
f.close()
file.close()

# In[87]:


result=sess.run(cnn.predictions, feed_dict={cnn.input_x:x_data ,cnn.dropout_keep_prob: 0.5})
print("##############################")
print("############결과값############")
print("##############################")
if(result==0):
    print("The output is 형사사건")
elif(result==1):
    print("The output is 민사사건")
else:
    print("The output is 행정사건")
    


