All code was tested on MATLAB 2015b

PROJ_DIR contains the project and all the .m files
Set the value of PROJ_DIR in main_svm_unsupervised.m

Download CIFAR 10 dataset from
https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR_DIR is the path in which the untared dataset is stored. 
Set the value of CIFAR_DIR in choose_data.m

Download vlfeat from 
http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz
Untar the file. Store the path in VLFEATROOT.
Set VLFEATROOT in features_compute.m
Note : Follow instructions from the website to install.

Get the minFunc from 
http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
Place it in PROJ_DIR

In order to perform Bag of Words Feature Extraction, run the files in this order
1) choose_data.m
2) features_compute.m
The features are thus extracted and stores in .mat files in PROJ_DIR.
To perform knn classification, run main_knn_cv.m
To perform SVM, run main_svm_L2.m

In order to perform Unsupervised Feature Extraction + L2 SVM, run the files in this order
1) choose_data.m
2) main_svm_unsupervised.m

In order to perform Unsupervised Feature Extraction + Knn, run the files in this order
1) choose_data.m
2) main_svm_unsupervised.m
3) main_knn_us.m