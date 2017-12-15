# Orientation in (Deep) Learning

A collection of information sources on deep learning with the focus on practical understanding of the technology and its usage. The challenge I try to face is to gain the necessary depth of understanding on the topic to be able to generalize proven solutions without going as deep as active researcher on deep learning need to be. The collection has a strong bias to supervised learning.

This collection is maintained by Michael Krämer ([@mkraemerx](https://twitter.com/mkraemerx)). If you think you can contribute to this list, please submit a pull request.

  0. [Basics](#basics)
  1. [Data Exploration & Preprocessing](#data-exploration)
  2. [Problem & Type Definition](#problem-definition)
  3. [Model Architecture](#architecture)
  4. [Training & Model Validation](#training-and-validation)

## <a name='basics'> Basics
* [Practical Deep Learning For Coders](http://course.fast.ai): A free online course that is meant to take 7 weeks to get you into practical usage of deep learning. The creators state that deep learning needs to get out of the ivory tower and present a course that requires almost no math.
* [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)
* [How Convolutional Neural Networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html): Understanding of what is happening inside of a deep learning network is quite a challenge. This blog describes a visual approach to understand the network and its layers.
* Matrices
  * [Properties of Matrices](http://fourier.eng.hmc.edu/e161/lectures/algebra/node2.html)
  * [Matrix Multiplication](https://www.khanacademy.org/math/precalculus/precalc-matrices/multiplying-matrices-by-matrices/v/matrix-multiplication-intro)
  * [Matrix Inverse](https://www.khanacademy.org/math/algebra-home/alg-matrices/alg-intro-to-matrix-inverses/v/inverse-matrix-part-1)
  * [Orthogonal Matrices](http://mathworld.wolfram.com/OrthogonalMatrix.html)
* Properties of Functions:
  * Continuous
  * [Monotonic Function](http://mathworld.wolfram.com/MonotonicFunction.html)
  * [Differentiable](https://en.wikipedia.org/wiki/Differentiable_function)
  * Non-negative
  * ...
* [Universal Approximator Theorem](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.6041&rep=rep1&type=pdf) (1993)
* [No Free Lunch Theorems](http://www.no-free-lunch.org/) by David Wolpert, William Macready

## <a name='data-exploration'> Data Exploration & Preprocessing
* [Introduction to Data Exploration](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/): Gives a good first overview about data exploration and cleanup. More targeted to classical machine learning, especially the sections about outliers and feature engineering should partly not be necessary in deep learning.
* [K-Means in SciKit-Learn](http://scikit-learn.org/stable/modules/clustering.html#k-means)
* [Dataset splitting in SciKit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* [Google Cloud Dataprep](https://cloud.google.com/dataprep/?hl=de): Tool in beta program that guesses data types and creates basic uni-variate properties and histograms. Interactive. Seems it has been evolved out of [DataWrangler](http://vis.stanford.edu/wrangler/).
* [Normalization in DeepLearning4J](http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/DataNormalization.htm)


## <a name='problem-definition'> Problem & Type Definition
* [Encoding categorical features](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features): Also know as One-hot-vectors or One-hot-encoding, this type of encoding performs way better than usage of ordinal types (numbers) to represent categories.
* Lower bound of needed observations for sparse, high-dimensional feature spaces is k for 2^k regions: [Non-Local Manifold Parzen Windows](http://www.cs.toronto.edu/~larocheh/publications/nlmp-nips-05.pdf) by Bengio Y., Larochelle H and Vincent P (in NIPS‘2005, MIT Press)

* [Curse of dimensionality](http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/)

## <a name='architecture'> Model Architecture
* [OpenCV](https://opencv.org/): Targeted to computer vision, ships with flexible recipes.
* [NLP_BestPractices](http://ruder.io/deep-learning-nlp-best-practices/): Best practices for text processing.
* [Keras](https://keras.io): Meta-API for definition of deep neuronal networks.
* [Estimator in Tensorflow](https://www.tensorflow.org/programmers_guide/estimators): High-level API for Tensorflow.
* [LossFunction](https://en.wikipedia.org/wiki/Loss_function): The function measure of the preciseness of predictions from the model.
* [Dataset Augmentation](https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/)

## <a name='training-and-validation'> Training & Model Validation
* [CrossValidation](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Back-Propagation](https://en.wikipedia.org/wiki/Backpropagation)
* [Train_DNN](http://rishy.github.io/ml/2017/01/05/how-to-train-your-dnn/)
