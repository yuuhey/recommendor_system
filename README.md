# recommendor_system
PIAI_project

<br/>

# About this project

💡 메뉴(상품) 추천 알고리즘 설계 및 개발(성능 목표치 : Hit Rate@K 0.55 이상)
</aside>

- 소비자(고객 id 기반)가 방문한 상권 내 먹거리 매장에 대한 상호명, 구매 상품(메뉴) 등의 업종 정보와 소비자가 구매한 상품의 평점 등의 데이터를 수집 및 분석
- 소비자의 성별, 연령 등의 정보를 기반으로 유사 소비자군을 군집화하고 해당 군집을 기반으로 메뉴 추천 시스템 개발
- 해당 상품 추천 엔진으로서는 사용자 기반 추천 시스템으로 CBF 기법을 활용하여 유사도를 활용한 소비자 간의 user-based similarity를 기반으로 메뉴 추천 수행
- 유사도가 높은 소비자의 평점 정보를 바탕으로 타깃 유저에 대한 개인화된 메뉴 추천 수행

<br/>

# **Problem Description**

- 구미 진평 먹자골목
- 가족단위 외식, 회식
- 나이, 성별 알 수 없음

<br/>
<br/>
<br/>

----
# **File Description**
## <span style="color:green" >[ 사용한 데이터 ]
- 3.소비자+ 유형별+선호+메뉴.csv
- MENU INFO KOREAN.csv
- MENU DSCRN INFO KOREAN.csv

<br/>

## <span style="color:green"> [ 파일 설명 ]

### 3.소비자+ 유형별+선호+메뉴.csv
[농림축산식품 공공데이터 포털](https://data.mafra.go.kr/opendata/data/indexOpenDataDetail.do?data_id=20141014000000000056&filter_ty=) 의 데이터로
건강식단추구/경제성추구/로컬푸드지향/식생활모험가/안전성중시 5가지 유형의 계절별 순위 메뉴
-> 사용자 유형 데이터와 사용자들의 데이터의 유사도를 분석하여 사용자의 유형을 분류하는 데에 사용

### chromedriver
크롤링 시 사용

### MENU INFO KOREAN.csv & MENU DSCRN INFO KOREAN.csv
[강원도 원주시_다국어메뉴정보](https://www.data.go.kr/data/15076727/fileData.do) & [강원도 원주시_다국어메뉴설명정보](https://www.data.go.kr/data/15099623/fileData.do)

강원도 원주시에 위치한 음식점에서 판매하는 메뉴를 한국어, 중국어, 영어, 일본어로 적어놓은 데이터
  
식당명, 메뉴명, 메뉴가격, 지역특산메뉴여부, 메뉴설명(주재료) 등의 컬럼으로 이루어진 데이터

### food_dataset_preprocessing.ipynb 
위 데이터를 메뉴에 대한 설명정보로 만들기 위해 전처리하는 코드작업

### menu_recipe_similarity.ipynb 
메뉴 설명, 대분류 등을 이용하여 워드임베딩을 통해 음식 간 유사도 분석 작업
  
**@ menu_recipe_similarity.py** 파일 내 정의함수를 사용하기 위해 py로 저장

### review_crawling.ipynb
음식 선호도 데이터를 생성하여 이용하기 위한 사용자 리뷰 크롤링 작업코드
네이버지도에서 임의로 리뷰가 많은 한 음식점을 선정하여 해당 음식점 리뷰의 리뷰어 중 10명의 리뷰를 크롤링

### main.ipynb
(1) 전처리
- 사용자의 카테고리 비율을 통해 선호도 분석
- 방문한 가게명과 리뷰 텍스트에서 메뉴 추출
- 사용자와 사용자 유형의 선호 음식 간 유사도를 계산하여 유형 분류
  
(2) 여러 가중치를 추가하여 최종 추천함수 구현

<br/>
<br/>

### @코드 작업 중 저장한 데이터프레임
#### raw user data
- 크롤링 작업 후 저장된 사용자 리뷰 데이터

#### menu ratio
- 사용자별 카테고리(한식/중식/일식/양식)에 대한 선호도 가중치
- 함수 내에서 사용자의 카테고리 선호도를 저장하고 전달하는 용도로 사용

#### new df
- 정제한 메뉴데이터셋 파일

#### simi_menu
- 음식 간 유사도 점수 데이터프레임

#### user_type_like
- 사용자 유형 음식 선호 데이터를 전처리 후 KNN으로 모든 음식에 대한 음식선호를 예측한 데이터프레임

#### user_url_df
- 크롤링 시 사용자의 리뷰데이터에 쉽게 접근하기 위한 url 저장

<br/>

----

## **[Notion Page](https://sleepy-judge-889.notion.site/a49e5ae824be467783b586db14cf541c)**

<img src="https://user-images.githubusercontent.com/118980404/225178675-dd56cae1-07ed-46a6-9934-78ab45432910.png" width="200" height="200"/>
