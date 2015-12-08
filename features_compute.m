clear all
clear workspace

VLFEATROOT = '/Users/samhitathakur/USC/Projects/EE660/vlfeat-0.9.20'

s = strcat(VLFEATROOT,'/toolbox/vl_setup');

run(s);

D = load ('reduced_data.mat');

data = D.ds.data;
trainY = D.ds.labels;

count = 0;

k = 100; %Number of codewords

pca_dim = 60;

NTrain = size(data,1); %Number of training samples

all_desc = [];


all_desc = dense_sift(D.ds);

fprintf('Computing the PCA vectors\n');
[signals,pca_dir] = do_pca(all_desc);

signals = signals(1:pca_dim,:);

%cluster
fprintf('Clustering vectors\n');
[idx,C] = kmeans(signals',k,'MaxIter',50);
clear signals;

fprintf('Building Bag of words...training images\n');
[hist] = build_bof(idx,k,NTrain);

save('trainX.mat','hist'); 
save('trainY.mat','trainY');
clear hist;
clear trainY;

%Load the test data
DTest = load('reduced_data_test.mat');

%Extract SIFT descriptors for the test data
test_desc = dense_sift(DTest.ds_test);
testY = DTest.ds_test.labels;

%Project test data along the pca dimensions of the training data
[rowTest,colTest] = size(test_desc);

mnTest = double(mean(test_desc',2));
data_test = double(test_desc') - repmat(mnTest,1,rowTest);

signalsTest = pca_dir * data_test;
signalsTest = signalsTest' ;%Make the rows as the data points
signalsTest = signalsTest(:,1:pca_dim);

%Find the closest cluster center (use the clusters found using the training
%samples
fprintf('Clustering test vectors\n');
idxTest = dsearchn(C,signalsTest);

%Build the bag of features
fprintf('Building Bag of words...test images\n');
NTest = size(testY,1);
histTest = build_bof(idxTest,k,NTest);

save('testX.mat','histTest');
save('testY.mat','testY');
