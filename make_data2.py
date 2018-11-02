
# coding: utf-8

# In[1]:


import os
import numpy as np
import codecs
Stable1=[]
Stable2=[]
Stable3=[]
Wtable1=[]
Wtable2=[]
Wtable3=[]
arrS1='./acceldata/Sdata/Hyeong/'
arrS2='./acceldata/Sdata/Min/'
arrS3='./acceldata/Sdata/Heng/'
arrW1='./acceldata/Wdata/Hyeong/'
arrW2='./acceldata/Wdata/Min/'
arrW3='./acceldata/Wdata/Heng/'
for i in range (9937):#9937 
    file_name = arrS1+str(i).zfill(5)+'.txt'
    file_name2= arrW1+str(i).zfill(5)+'.txt'
    fS = open(file_name,'r',-1,encoding = 'UTF8')
    fL = open(file_name2,'r',-1,encoding ='UTF8')
    line = fS.read()
    line2= fL.read()
    Slist=[]
    Wlist=[]
    Slist= line.split('\n')
    Wlist= line2.split('\n')
    del Slist[len(Slist)-1]#마지막 두개 제거
    del Slist[len(Slist)-1]
    del Wlist[len(Wlist)-1]
    del Wlist[len(Wlist)-1]
    
    
    Stable1.append(Slist)
    Wtable1.append(Wlist)
    fS.close()
    fL.close()
for i in range (26228):#26228 
    file_name = arrS2+str(i).zfill(5)+'.txt'
    file_name2= arrW2+str(i).zfill(5)+'.txt'
    fS = open(file_name,'r',-1,encoding = 'UTF8')
    fL = open(file_name2,'r',-1,encoding ='UTF8')
    line = fS.read()
    line2= fL.read()
    Slist=[]
    Wlist=[]
    Slist= line.split('\n')
    Wlist= line2.split('\n')
    del Slist[len(Slist)-1]#마지막 두개 제거
    del Slist[len(Slist)-1]
    del Wlist[len(Wlist)-1]
    del Wlist[len(Wlist)-1]
    
    Stable2.append(Slist)
    Wtable2.append(Wlist)
    fS.close()
    fL.close()
for i in range (13706):#13706
    file_name = arrS3+str(i).zfill(5)+'.txt'
    file_name2= arrW3+str(i).zfill(5)+'.txt'
    fS = open(file_name,'r',-1,encoding = 'UTF8')
    fL = open(file_name2,'r',-1,encoding ='UTF8')
    line = fS.read()
    line2= fL.read()
    Slist=[]
    Wlist=[]
    Slist= line.split('\n')
    Wlist= line2.split('\n')
    del Slist[len(Slist)-1]#마지막 두개 제거
    del Slist[len(Slist)-1]
    del Wlist[len(Wlist)-1]
    del Wlist[len(Wlist)-1]
    
    Stable3.append(Slist)
    Wtable3.append(Wlist)
    fS.close()
    fL.close() 


# In[2]:


TopWordHy=[]
TopWordMin=[]
TopWordHe=[]
for i in range(9937):#9937
    TW=[0]*10 #top word
    TS=[0]*10 #top similar
    if( i == 9807):
        continue; # 잘못패치되는 경우 에러
    for j in range(len(Stable1[i])):#한 문장 단어개수 10개 추리기
        if( Stable1[i][j]< '1'):
            for k in range(10):#소팅시키는작업 
                if(str(TS[k])<Stable1[i][j]):#만약 더 큰숫자가 나타난다면
                    for l in range(10 - k):# 맨뒤부터 차례대로 앞숫자 댕겨와서
                        if(k<9):
                            TW[9-l]= TW[8-l]
                            TS[9-l]= TS[8-l]#순서대로 워드 밀어주는 방법
                    TW[k]=Wtable1[i][j]
                    TS[k]=Stable1[i][j]
                    break;#한번 걸리면 바로 브레이크걸어준다 
    #print(TW)
    TopWordHy.append(TW)# 다추리면 10* len으로  리스트화
    
for i in range(26228):#26228
    if ( i == 10597):
        continue;
    elif(i == 17361):#10개미만 문서 제거
        continue;
    TW=[0]*10 #top word
    TS=[0]*10 #top similar
    for j in range(len(Stable2[i])):
        if( Stable2[i][j]< '1'):
            for k in range(10):
                if(str(TS[k])<Stable2[i][j]):
                    for l in range(10 - k):
                        if(k<9):
                            TW[9-l]= TW[8-l]
                            TS[9-l]= TS[8-l] 
                    TW[k]=Wtable2[i][j]
                    TS[k]=Stable2[i][j]
                    break;
    TopWordMin.append(TW)
for i in range(13706):#13706
    if(i== 10686):
        continue;
    elif(i== 12937):# 10개 미만 문서 제거
        continue;    
    TW=[0]*10 #top word
    TS=[0]*10 #top similar
    for j in range(len(Stable3[i])):
        if( Stable3[i][j]< '1'):
            for k in range(10):
                if(str(TS[k])<Stable3[i][j]):
                    for l in range(10 - k):
                        if(k<9):
                            TW[9-l]= TW[8-l]
                            TS[9-l]= TS[8-l] 
                    TW[k]=Wtable3[i][j]
                    TS[k]=Stable3[i][j]
                    break;
    TopWordHe.append(TW)


# In[3]:


print("단어 10개미만의 형사문서 번호")
for i in range(9936):
    for j in range(10):
        if(TopWordHy[i][j]== 0):
            print(i)
print("단어 10개 미만의 민사문서 번호")
for i in range(26226):
    for j in range(10):
        if(TopWordMin[i][j]== 0):
            print(i)
print("단어 10개 미만의 행정문서 번호")
for i in range(13704):
    for j in range(10):
        if(TopWordHe[i][j]== 0):
            print(i)


# In[4]:


ar1='./Top10Data/Hyeong/'
ar2='./Top10Data/Min/'
ar3='./Top10Data/Heng/'
i=0
for i in range (9936):#9936
    filename= ar1+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')
    for j in range(len(TopWordHy[i])):
        output.write(TopWordHy[i][j]+'\n')#output file에다 저장
    output.close()
for i in range (26226):#26226
    filename= ar2+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')
    for j in range(len(TopWordMin[i])):        
        output.write(TopWordMin[i][j]+'\n')
    output.close()
for i in range (13704):#13704
    filename= ar3+str(i).zfill(5)+'.txt'
    output = open(filename,'w',-1,encoding = 'UTF8')   
    for j in range(len(TopWordHe[i])):        
        output.write(TopWordHe[i][j]+'\n')
    output.close()


# In[5]:


arr1='./Top10Data/Hyeong/'
arr2='./Top10Data/Min/'
arr3='./Top10Data/Heng/'
output = open('./data.csv','w',-1,encoding = 'UTF8')
output.write('고시 목적 계획 기준 공공사업 사항 이용 시행 무관 규정,Administrative\n')
#맨 첫줄은 쓰레기값을 줌, 왜냐하면 첫줄부터 읽지않는 에러가 나기때문
for i in range (26226):
    if(i < 9936):#9936
        file_name1 = arr1+str(i).zfill(5)+'.txt'
        f1 = open(file_name1,'r',-1,encoding = 'UTF8')
        line = f1.read()
        lists= line.split('\n')
        del lists[len(lists)-1]#마지막 한개 제거
        for j in range(10):
            output.write(lists[j])
            if(j<9):
                output.write(' ')
        output.write(',Criminal'+'\n')
        f1.close()
    if(i<26226):
        file_name2 = arr2+str(i).zfill(5)+'.txt'
        f2 = open(file_name2,'r',-1,encoding = 'UTF8')
        line = f2.read()
        lists= line.split('\n')
        del lists[len(lists)-1]#마지막 한개 제거
        for j in range(10):
            output.write(lists[j])
            if(j<9):
                output.write(' ')
        output.write(',Civil'+'\n')
        f2.close()
    if (i < 12304):#13704
        file_name3 = arr3+str(i).zfill(5)+'.txt'
        f3 = open(file_name3,'r',-1,encoding = 'UTF8')
        line = f3.read()
        lists= line.split('\n')
        del lists[len(lists)-1]#마지막 한개 제거
        for j in range(10):
            output.write(lists[j])
            if(j<9):
                output.write(' ')
        output.write(',Administrative'+'\n')
        f3.close()
output.close()
print("data.csv 만들기 완료")

