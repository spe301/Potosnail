# Potosnail

<img src="Images/download.jpg/">
This is a collection of helper functions that can accelerate Machine Learning and allow me to move much faster

author: Spencer Holley

contact: aacjpw@gmail.com

# Installation and import 

the latest version can be installed via pip install Potosnail==0.2.1
However while I work to resolve bugs, use 0.0.5 if you need to do a keras gridsearch, and 0.0.9 if you need to use Smote sampling

# Tutorials and Case Studies
Check out the notebooks in the tutorials folder to see some implementation. If you want to see more real-world use cases for potosnail, chech out the Case_Studies folder

## Why do these model building functions take in 10 arguments?
<img src="Images/4tqs0v.jpg/">
If you use DeepTabularRegression, DeepTabularClassification, RNN, or CNN from DeepLearning you'll notice that these funtions require 10 arguments, this is so that these functions work well with the CollectPerformance function, this is because I had to code a gridsearch from scratch and there needed to be a fixed number of for loops for a fixed number of parameters. if you don't like this you can use FastNN (for the DeepTabulars), FastRNN, and FastCNN instead, you can run a gridsearch on them with DeepLearning().TestDl().
