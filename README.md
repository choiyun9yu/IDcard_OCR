# IDcard_OCR
Picture Taken  
Face Detaction -> Landmark Detaction -> Face Aligment -> Featture Extraction -> Feature Matching  
Text Detaction -> Text Recognizer -> Text Extraction -> Text Matching

## Update
**Feb 28, 2022** : 오픈소스 모델들로 OCR, Face Verification 테스트

## Getting Started
### Dependency
- 가상환경 : python 3.8
- requirements : requirements.txt

### Data set
- data/

### Open API
- Pytesseract   
pip install pytesseract

- PaddleOCR  
pip install paddelpaddle  
pip install paddleocr  

- face_recognition
pip install cmake
pip install dlib

- Deepface
pip install deepface

## Face Recognition
얼굴과 딥러닝에 관련된 기술은 크게 두 가지로 나눌 수 있다.  
사진이나 동영상에서 얼굴의 위치를 찾는 얼굴 검출(Face Detaction) 기술과  
해당 얼굴이 동일인 여부, 특징 등을 추출해내는 얼굴 인식(Face Recognition) 기술이다.  
      
여기서 얼굴 인식 기술은 다시 2가지로 나뉜다.
하나는 1:1 검증으로 두 장의 사진이 동일인 인지 파악하는 얼굴 검증(Face Verification),
다른 하나는 1:N 검증으로 새로 들어온 한 장의 사진 속 사람이 DB에 존재하는  
인물인지 검증하는 얼굴 식별(Face Identification) 이다. 


## Evaluation
### 얼굴인식 기술 
- 얼굴 인식은 비교하 두 이미지를 먼저 Backbone Network에 넣고,  
  최종단에 나오는 Embedding Vector 두 개를 비교하여 동일인 여부를 결정한다.
- Backbone Network는 보통 CNN구조이며, Resnet, Inception, VGG 등 여러가지 네트워크가 될 수 있다.
- Embedding Vector는 네으쿼으 최종단에 있는 길이가 N인 벡터이다.
- 요약하자면 얼굴이미지를 1차원 벡터로 정보를 압축하여 두 백터의 유사도를 비교하는 것이다.

#### 유사도 비교 방법 2가지
두 가지 방법은 학습을 어떤 Loss로 했느냐에 따라 결정된다.
- 1. 두 벡터의 L2 거리를 직접 구하는 방법(유클리드 거리)
triplet-loss로 학습한 경우
- 2. 두 벡터의 Cosine 거리를 구해서 구하는 방법   
softmax-loss로 학습한 경우
- 참조 : https://m.blog.naver.com/laonple/221583231520
