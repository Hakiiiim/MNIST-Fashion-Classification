# MNIST-Fashion-Classification

Classification of a sample of 6000 images from the mnist-fashion zalando database using first image processing methods and then a convolutional neural network.

The work is done by me and my colleague Younes.

The file script.py is the main python file, it is entirely commented in english and well organised: In the first part it uses 7 classification models (Logistic regression, Decision tree, Gradient descent, Gradient boosting, KNN, SVM, shallow neural network) on a DataFrame of geomtric descriptors (features) exctracted using regionproprs function from the scikit-image library in python.  

In the second part, we use the library Tensorflow in order to implment a convolutional neural network with one 32 convolutional layer and Two Dense layers with a dropout of 0.25, this second method gives widely better results reaching an accuracy of 92%.

The accuracy we could achieve using the geometric descriptors is not very good 60%, this is due to the information we lose at the moment we binarize the images.

The report written using a Jupyter Notebook is in Frensh, it is the same as the pdf file rapport.pdf, it shows the important plots and images visualizations we have done in order to better communicate our results.

Thank you for reading ^^
