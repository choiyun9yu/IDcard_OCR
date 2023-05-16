# Face Verification between IDcard and Face photo
[![arcface](https://github.com/serengil/deepface)

## Overview
In deep learning, facial-related technologies are largely divied into two categories.
Face Detection, which finds the location of the face in a picture or video,
and Face Recognition that determines wheter the face is the same.
  
Additionally, Face Recognition is again divided into two categories. The first one is Face Verification, which determines whether two facial photos belong to the same person through a 1:1 verification process.
The second category is Face Identification, which involves determining whether a newly captured facial photo matches any person in an existing database through a 1:N verification process.

This project aims to develop a deep learning model for Face Verification, where it compares an identification photo with a facial image to determine if they belong to the same person.


## Face Verification Model Structure

![Structure](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/Face%20Verifiaction%20Model%20Structure.png)

First, the Face Verification model detects faces when two photos are provided as input.
After detecting the face, the feature is extracted by putting it in a pre-trained backbone network to extract the feature. To extract features more effectively, various preprocessing techniques such as aligning the faces can be applied at this stage.

After the preprocessing and backbone, the Embedding Vector of each face photo is extracted.
The difference in distance between these vectors determines whether the two faces are the same. If the distance difference exceeds the threshold, it is judged to be non-identical, and if it does not exceed the threshold, it is judged to be the same person.


## Contents
We used ArcFace open-source because it was the most accurate among open-source models  in our dataset.

|Model|Acc|
|-----|---|
|ArcFace|**0.86**|
|VGGFace|0.84|
|DeepFace|0.60|
|FaceNet|0.79|

In this project, we used a method to increase accuracy by optimizing the threshold.
![Threshold](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/Threshold%20Train.png)

*You can find related information through ![Threshold_Train.ipynb](https://github.com/choiyun9yu/pr.FaceVerification/blob/main/img/Face%20Verifiaction%20Model%20Structure.png).*
