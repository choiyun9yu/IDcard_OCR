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
- 비공개

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
얼굴인식 기술 성능 평가 지표
