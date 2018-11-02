
# coding: utf-8

# In[1]:


import os
import numpy as np
import gensim
import codecs
from konlpy.tag import Twitter
from konlpy.utils import pprint
import collections
import time
import xlrd
import xlwt
import re
from gensim.models import Word2Vec


# In[2]:


arr='./legaldata/Min/'
arr1='./legaldata/Hyeong/'
arr2='./legaldata/Heng/'
lines1= []
table1=[]
table2=[]
table3=[]
for i in range (26329):#26329
    file_name = arr+str(i).zfill(5)+'.txt'
    try:
        f = open(file_name,'r',-1,encoding = 'UTF8') 
    except FileNotFoundError as e:
        pass
    else:
        if(f):
            line = f.read()
            lines = []
            lines = line.split('\n')
            lines1.append(lines)
            table2.append(lines)
        else:
            print('error read file')
    f.close()
for i in range (10171):#10171
    file_name = arr1+str(i).zfill(5)+'.txt'
    try:
        f = open(file_name,'r',-1,encoding = 'UTF8') 
    except FileNotFoundError as e:
        pass
    else:
        if(f):
            line = f.read()
            lines = []
            lines = line.split('\n')
            lines1.append(lines)
            table1.append(lines)
        else:
            print('error read file')
    f.close()
for i in range (13735):#13735
    file_name = arr2+str(i).zfill(5)+'.txt'
    try:
        f = open(file_name,'r',-1,encoding = 'UTF8') 
    except FileNotFoundError as e:
        pass
    else:
        if(f):
            line = f.read()
            lines = []
            lines = line.split('\n')
            lines1.append(lines)
            table3.append(lines)
        else:
            print('error read file')
    f.close()    


# In[12]:


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
for i in range(10):# 10개씩 3카테고리의 샘플파일도 Word2vec을위한 tokenize
    file_name1 = './data/y'+str(i+1)+'.txt'
    file_name2 = './data/e'+str(i+1)+'.txt'
    file_name3 = './data/m'+str(i+1)+'.txt'
    file1 = open(file_name1,'r',-1,encoding='UTF8')
    for j in range(300):
        text=file1.readline()
        if(text=="【이 유】\n"):
            break;
    f=file1.read()
    flist1=get_list(f)
    lines1.append(flist1)
    file1.close()
    file2 = open(file_name2,'r',-1,encoding='UTF8')
    for j in range(300):
        text=file2.readline()
        if(text=="【이 유】\n"):
            break;
    f=file2.read()
    flist2=get_list(f)
    lines1.append(flist2)
    file2.close()
    file3 = open(file_name3,'r',-1,encoding='UTF8')
    for j in range(300):
        text=file3.readline()
        if(text=="【이 유】\n"):
            break;
    f=file3.read()
    flist3=get_list(f)
    lines1.append(flist3)
    file3.close()


# In[14]:


embedding_model = Word2Vec(lines1,size=100, window=5, min_count=0, workers=4,iter=100 , sg=0)


# In[27]:


word_vectors= embedding_model.wv


# In[28]:


word_vectors.save('./union_model_Noverlap')


# In[3]:


model=gensim.models.KeyedVectors.load('./union_model_Noverlap')#모델을 읽고


# In[4]:


print("Word2Vec 학습 완료 & 학습모델 저장")


# In[5]:


print("형사 most_similar 테이블 선정")
word1=[]
temp=model.most_similar("형사",topn=100)
for i in range(len(temp)):
    word1.append(temp[i][0])
print("민사 most_similar 테이블 선정")
word2=[]
temp=model.most_similar("민사",topn=100)
i=0
for i in range(len(temp)):
    word2.append(temp[i][0])
print("행정 most similar 단어 테이블 선정")
word3=[]
temp=model.most_similar("행정",topn=100)
i=0
for i in range(len(temp)):
    word3.append(temp[i][0])
# 각 카테고리에 해당하는 mostsimilar_100 리스트 가장 가까운순부터


# In[25]:


ar1='./acceldata/Sdata/Hyeong/'
ar2='./acceldata/Sdata/Min/'
ar3='./acceldata/Sdata/Heng/'
arr1='./acceldata/Wdata/Hyeong/'
arr2='./acceldata/Wdata/Min/'
arr3='./acceldata/Wdata/Heng/'
i=0
for i in range (9937):#9937
    filename= ar1+str(i).zfill(5)+'.txt'
    filename2= arr1+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')
    output2 = open(filename2,'w',-1,encoding = 'UTF8')
    for j in range(len(valuelist1[i])):
        out= str(valuelist1[i][j])+'\n'
        output.write(out)#output file에다 저장
        output2.write(table1[i][j]+'\n')#output file에다 저장
    output.close()
    output2.close()
for i in range (26228):#26228
    filename= ar2+str(i).zfill(5)+'.txt'
    filename2= arr2+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')
    output2 = open(filename2,'w',-1,encoding = 'UTF8')
    for j in range(len(valuelist2[i])):
        out= str(valuelist2[i][j])+'\n'
        output.write(out)#output file에다 저장
        output2.write(table2[i][j]+'\n')#output file에다 저장
    output.close()
    output2.close()
for i in range (13706):#13706
    filename= ar3+str(i).zfill(5)+'.txt'
    filename2= arr3+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')
    output2 = open(filename2,'w',-1,encoding = 'UTF8')
    for j in range(len(valuelist3[i])):
        out= str(valuelist3[i][j])+'\n'
        output.write(out)#output file에다 저장
        output2.write(table3[i][j]+'\n')#output file에다 저장
    output.close()
    output2.close()

