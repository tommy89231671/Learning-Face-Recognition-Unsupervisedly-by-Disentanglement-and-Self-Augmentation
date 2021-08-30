# Project
Dataset:
https://drive.google.com/file/d/1TDphufJnVbx5jBBPI1qWPzqePlpcnjoh/view?usp=sharing

Publication:
[Learning Face Recognition Unsupervisedly by Disentanglement and Self Augmentation
](https://github.com/tommy89231671/Learning-Face-Recognition-Unsupervisedly-by-Disentanglement-and-Self-Augmentation/blob/master/Learning%20Face%20Recognition%20Unsupervisedly%20by%20Disentanglement%20and%20Self-Augmentation-1.jpg)

Purpose:
  
  We would like to classsify few people in fix group like family precisely.And do well on different situation.For example,whether a man wears glassese or not,we can say that he is the same person. Another example is no matter the camera's vision angle is, the model should not be influenced.
  

Input:

1.Input image from videos'frame

2.Positive and Negative pairs

 (1)Take two images into a pre-trained face feature extraction model(Ex:lightCNN) and use knn to find positive pairs.
 
 (2)Use frame's property which is that the people in the same frame will be the different person and it is the negative pairs.


Description:

Input1 on VAE+InfoGAN.

Input2 on Metric Learning and Classifier.

Model Graph:

![image]( https://github.com/tommy89231671/Project/blob/Add-classifier/Model%20for%20project.jpg)

![image]( https://github.com/tommy89231671/Project/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%B5%90%E5%90%88%E6%99%BA%E6%85%A7%E5%AE%B6%E5%BA%AD%E7%9B%A3%E6%8E%A7%E6%B5%B7%E5%A0%B1.jpg
)

 
 
