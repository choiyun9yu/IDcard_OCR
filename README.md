# Face Verification between IDcard and Face photo
[![arcface](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/deepface.svg)](https://github.com/serengil/deepface)

## Overview
In the field of deep learning, facial-related technologies can be broadly categorized into two main groups: Face Detaction and Face Recogniction.
The Face Detection finds the location of the face in a picture or video
and Face Recognition determines wheter the face is the same.
  
Face Recognition can be further divided into two subcategories. The first one is Face Verification, which utilizes a 1:1 verification process to establish whether two facial photos correspond to the same person.
The second category is Face Identification, which involves determining whether a newly captured facial photo matches any person in an existing database through a 1:N verification process.

The objective of this project is to develop a deep learning model for Face Verification. This model compares an identification photo with a real-time facial image to determine if they belong to the same person.

## Face Verification Model Structure

![Structure](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/Face%20Verifiaction%20Model%20Structure.png)

Initially, the Face Verification model detects faces when two photos are provided as input.
Once the face is detected, the features are extracted by passing it through a pre-trained backbone network. To extract features more effectively, this model can perform various preprocessing techniques such as aligning the faces can be applied at this stage.

After the backone network and preprocessing stages, the Embedding Vector of each face photo is extracted. The difference in distance between these vectors is used to determine whether the two faces are the same. If the distance difference exceeds the threshold, it is judged to be non-identical, and if it does not exceed the threshold, it is judged to be the same person.


## Contents
In some cases, individuals attempt to deceive the identification process in places like convenience stores or bars.
Therefore, the aim of this project is to create a model that can accurately determine if the ID-card holder and the person presenting the ID are the same.

![Threshold_Train.ipynb](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/img.png)

I chose to use the ArcFace open-source model because it demonstrated the highest accuracy among the available open-source models in our datasets.

|Model|Acc|
|-----|---|
|ArcFace|**0.86**|
|VGGFace|0.84|
|DeepFace|0.60|
|FaceNet|0.79|

Additionally, in this project, threshold optimization was performed to enhance the accuracy of the existing model.

![Threshold](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/Threshold%20Train.png)

*You can find related code through [Threshold_Train.ipynb](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/Threshold_Train.ipynb).*
