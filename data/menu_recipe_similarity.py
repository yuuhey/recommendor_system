#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[2]:


# 음식, 음식점 데이터 merge
import pandas as pd
import numpy as np

menu = pd.read_csv("new_df.csv")
menu


# In[3]:


list(menu['메뉴설명(MENU_DSCRN)'])


# ### 메뉴재료로만 유사도 계산
# 
# **Scikit-Learn 의 문서 전처리 기능**
# 
# BOW (Bag of Words) 문서를 숫자 벡터로 변환하는 가장 기본적인 방법은 BOW (Bag of Words)이다. BOW 방법에서는 전체 문서 {d1,d2,…,dn} 를 구성하는 고정된 단어장(vocabulary) {t1,t2,…,tm}를 만들고 di라는 개별 문서에 단어장에 해당하는 단어들이 포함되어 있는지를 표시하는 방법이다.
# 
# xi,j=문서 di내의 단어 tj의 출현 빈도
# 
# 또는
# 
# xi,j = 0 : 만약 단어 tj가 문서 di 안에 없으면 1 : 만약 단어 tj가 문서 di 안에 있으면
# 
# TfidfVectorizer : CountVectorizer와 비슷하지만 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 벡터를 만든다. -> TF-IDF(Term Frequency – Inverse Document Frequency) 인코딩은 단어를 갯수 그대로 카운트하지 않고 모든 문서에 공통적으로 들어있는 단어의 경우 문서 구별 능력이 떨어진다고 보아 가중치를 축소하는 방법이다.

# In[4]:


# n-그램:단어장 생성에 사용할 토큰의 크기를 결정한다. 모노그램(1-그램)은 토큰 하나만 단어로 사용하며 바이그램(2-그램)은 두 개의 연결된 토큰을 하나의 단어로 사용한다.
# Stop Words:문서에서 단어장을 생성할 때 무시할 수 있는 단어를 말한다. 보통 영어의 관사나 접속사, 한국어의 조사 등이 여기에 해당한다. stop_words 인수로 조절할 수 있다.
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
tfidf_matrix = tf.fit_transform(menu['메뉴설명(MENU_DSCRN)'])


# In[5]:


print(tfidf_matrix[10])


# In[6]:


tfidf_matrix.shape


# 코사인 유사도(Cosine Similarity)
# 코사인 유사도(Cosine Similarity)을 사용하여 두 영화 사이의 유사성을 나타내는 숫자 수량을 계산할 것입니다. 수학적으로 다음과 같이 정의됩니다.
# 
# cosine(x,y)=x.y⊺||x||.||y||
#  
# TF-IDF 벡터 라이저를 사용 했으므로 Dot Product를 계산하면 코사인 유사도 점수를 직접 얻을 수 있습니다. 따라서 cosine_similarities 대신 sklearn의 linear_kernel을 사용하는 것이 훨씬 빠릅니다.

# In[7]:


# linear_kernel는 두 벡터의 dot product 이다.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[8]:


menu_cates = menu['메뉴카테고리소분류명']
indices = pd.Series(menu.index, index=menu['메뉴카테고리소분류명'])

print(menu_cates.head(), indices.head())


# In[9]:


def get_recommendations(menu_cate):
    idx = indices[menu_cate]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    menu_indices = [i[0] for i in sim_scores]
    return menu_cates.iloc[menu_indices][:5]


# In[10]:


get_recommendations('치킨')


# In[11]:


display(menu[menu['메뉴카테고리소분류명']=='추어탕'])
display(menu[menu['메뉴카테고리소분류명']=='주먹밥'])


# In[12]:


display(menu[menu['메뉴카테고리소분류명']=='닭갈비'])
display(menu[menu['메뉴카테고리소분류명']=='볶음탕'])
display(menu[menu['메뉴카테고리소분류명']=='깐풍기'])
display(menu[menu['메뉴카테고리소분류명']=='라조육'])


# ### 대분류명, 소분류명도 이용하여 유사도 계산

# In[13]:


menu['soup'] = menu['메뉴설명(MENU_DSCRN)'] + "  " + menu['메뉴카테고리대분류명'] + "  " + menu['메뉴카테고리소분류명']
menu['soup']


# In[14]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(menu['soup'])


# In[15]:


sorted(tf.vocabulary_.items())


# In[16]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim


# In[17]:


menu_cates = menu['메뉴카테고리소분류명']
indices = pd.Series(menu.index, index=menu['메뉴카테고리소분류명'])


# In[18]:


get_recommendations('소갈비').head(3)


# In[19]:


display(menu[menu['메뉴카테고리소분류명']=='갈비'])
display(menu[menu['메뉴카테고리소분류명']=='갈비찜'])
display(menu[menu['메뉴카테고리소분류명']=='스테이크'])
display(menu[menu['메뉴카테고리소분류명']=='생굴'])


# In[20]:


get_recommendations('닭갈비').head(3)


# In[21]:


display(menu[menu['메뉴카테고리소분류명']=='닭갈비'])
display(menu[menu['메뉴카테고리소분류명']=='닭볶음탕'])
display(menu[menu['메뉴카테고리소분류명']=='깐풍기'])
display(menu[menu['메뉴카테고리소분류명']=='주물럭'])


# In[22]:


get_recommendations('티라미수')


# In[23]:


display(menu[menu['메뉴카테고리소분류명']=='티라미수'])
display(menu[menu['메뉴카테고리소분류명']=='크로플'])
display(menu[menu['메뉴카테고리소분류명']=='츄러스'])
display(menu[menu['메뉴카테고리소분류명']=='와플'])


# In[24]:


get_recommendations('치즈돈까스').head(3)


# In[25]:


get_recommendations('칼국수').head(3)


# In[26]:


display(menu[menu['메뉴카테고리소분류명']=='계란탕'])
display(menu[menu['메뉴카테고리소분류명']=='계란찜'])
display(menu[menu['메뉴카테고리소분류명']=='오므라이스'])


# In[27]:


display(menu[menu['메뉴카테고리소분류명']=='계란탕'])


# ## 가격 추가

# for i in cosine_sim:
#     for j in i:
#         if j<1:
#             print(j)

# 메뉴명, 메뉴재료, 메뉴카테고리로 워드임베딩 후 유사도 계산시 유사도 0인 아이템이 많이 나오는 문제가 있음

# In[28]:


import matplotlib.pyplot as plt
menu['메뉴가격'].hist()


# 적당한 비닝과 유사도 계산을 위한 정규화 필요

# In[29]:


menu['메뉴가격'].value_counts()


# In[30]:


menu['메뉴가격']


# In[31]:


# 최대최소정규화를 통해 0부터 1까지의 값으로 스케일링
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler()
transformer.fit(menu[['메뉴가격']])
# print(transformer.data_min_)
# print(transformer.data_max_)
scale_cost = transformer.transform(menu[['메뉴가격']]).reshape(-1,)
print(scale_cost)


# In[32]:


menu[['메뉴가격']].메뉴가격


# In[33]:


# 메뉴가격 차를 이용하여 가격 고려
mat1 = np.repeat([scale_cost], len(menu['메뉴가격']) , axis=0)
mat2 = mat1.T
cost_sim = np.ones((len(menu['메뉴가격']), len(menu['메뉴가격']))) - abs(mat1-mat2)
cost_sim


# In[34]:


menu_simi_co = (cosine_sim * 1 # 1. 텍스트 유사도
                 + cost_sim * 0.01  # 2. 가격
                 )

menu_simi_co_sorted_ind = menu_simi_co.argsort()[:, ::-1]

# 최종 구현 함수
def find_simi_menu(menu_name):
    
    idx = indices[menu_name]
    sim_scores = list(enumerate(menu_simi_co[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    menu_indices = [i[0] for i in sim_scores]
    return menu_cates.iloc[menu_indices][:5]


# In[35]:


find_simi_menu('닭갈비')


# In[36]:


display(menu[menu['메뉴카테고리소분류명']=='닭갈비'])
display(menu[menu['메뉴카테고리소분류명']=='깐풍기'])
display(menu[menu['메뉴카테고리소분류명']=='주물럭'])


# In[37]:


menu[menu['메뉴카테고리소분류명']=='닭갈비']


# In[38]:


menu[menu['메뉴카테고리소분류명']=='통닭']


# In[39]:


menu_simi_co = (cosine_sim * 1 # 1. 텍스트 유사도
                 + cost_sim * 0.01  # 2. 가격
                 )

menu_simi_co_sorted_ind = menu_simi_co.argsort()[:, ::-1]

# 최종 구현 함수
def find_simi_menu(menu_name):
    
    idx = indices[menu_name]
    sim_scores = list(enumerate(menu_simi_co[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    menu_indices = [i[0] for i in sim_scores]
    return menu_cates.iloc[menu_indices][:5]


# ## 카테고리 가중치 추가

# In[40]:


find_simi_menu('닭갈비')


# In[110]:


menu_simi_co = (cosine_sim * 1 # 1. 텍스트 유사도
                 + cost_sim * 0.01  # 2. 가격
                 )

menu_simi_co_sorted_ind = menu_simi_co.argsort()[:, ::-1]

# 최종 구현 함수
def find_simi_menu(menu_name):
    
    idx = indices[menu_name]    
    
    #사용자에 대한 메뉴 가중치 파일 받아오기
    menu_ratio = pd.read_csv('menu_ratio', index_col=0)
    
    a = list(menu['메뉴카테고리대분류명'])
    replacements = {'한식':float(menu_ratio.loc['한식']),
                   '제과류':float(menu_ratio.loc['카페']),
                   '양식':float(menu_ratio.loc['양식']),
                   '아시아/퓨전 음식':float(menu_ratio.loc['아시아/퓨전 음식']),
                   '일식':float(menu_ratio.loc['일식']),
                   '패스트푸드':float(menu_ratio.loc['분식/치킨']),
                   '중식':float(menu_ratio.loc['중식']),
                   '기타':0}
    replacer = replacements.get
    
    w = 1 #카테고리 가중치 설정
    #카테고리 선호비율을 축소하고 최종 추천된 메뉴점수에 더하기
    menu_simi = menu_simi_co[idx]+ [x*y for x,y in zip([replacer(n, n) for n in a],[w]*len(a))]
    sim_scores = list(enumerate(menu_simi))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    menu_indices = [i[0] for i in sim_scores]
    return menu_cates.iloc[menu_indices][:5]


# ## 사용자유형 가중치 추가

# In[146]:


menu_simi_co = (cosine_sim * 1 # 1. 텍스트 유사도
                 + cost_sim * 0.01  # 2. 가격
                 )

menu_simi_co_sorted_ind = menu_simi_co.argsort()[:, ::-1]

# 입력변수가 하나 더 생겨서 파일 한꺼번에 돌릴시 중간에러를 방지하기 위해 함수이름을 바꿉니다..
def find_simi_menu_ver3(menu_name,w): #여기서 w는 사용자유형에 대한 가중치
    
    idx = indices[menu_name]    
    
    #사용자에 대한 메뉴 가중치 파일 받아오기
    menu_ratio = pd.read_csv('menu_ratio', index_col=0)
    
    a = list(menu['메뉴카테고리대분류명'])
    replacements = {'한식':float(menu_ratio.loc['한식']),
                   '제과류':float(menu_ratio.loc['카페']),
                   '양식':float(menu_ratio.loc['양식']),
                   '아시아/퓨전 음식':float(menu_ratio.loc['아시아/퓨전 음식']),
                   '일식':float(menu_ratio.loc['일식']),
                   '패스트푸드':float(menu_ratio.loc['분식/치킨']),
                   '중식':float(menu_ratio.loc['중식']),
                   '기타':0}
    replacer = replacements.get
    cate_w = 1 #카테고리 가중치 설정
    
    # 사용자 유형의 메뉴선호 반영
    user_type_like = pd.read_csv('user_type_like', index_col=0)
    
    menu_simi = menu_simi_co[idx]+ [x*y for x,y in zip([replacer(n, n) for n in a],[cate_w]*len(a))] + user_type_like.loc[w[0]].values *(0.01)*w[1]
    sim_scores = list(enumerate(menu_simi))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    menu_indices = [i[0] for i in sim_scores]
    return menu_cates.iloc[menu_indices][:5]


# In[144]:


#확인용
find_simi_menu_ver3('닭갈비',['건강식단추구',0])


# In[143]:


#확인용
find_simi_menu_ver3('닭갈비',['경제성추구',0.615353])


# ## 사용자별 메뉴 결측치 빈도수 채우기 위한 음식 유사도 행렬

# In[45]:


menu_simi_co


# In[47]:


transformer = MinMaxScaler()
transformer.fit(menu_simi_co)
simi_menu = transformer.transform(menu_simi_co)
simi_menu = pd.DataFrame(simi_menu)
simi_menu


# 뭐지 왜 대칭 아니지

# In[44]:


# 다른 파일에서도 쓰려고 저장하기
simi_menu.to_csv('simi_menu')

