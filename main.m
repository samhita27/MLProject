clear all
clear workspace

VLFEATROOT = '/Users/samhitathakur/USC/Projects/EE660/vlfeat-0.9.20'

s = strcat(VLFEATROOT,'/toolbox/vl_setup');

run(s);

D1 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_1.mat')
D2 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_2.mat')
D3 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_3.mat')
D4 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_4.mat')
D5 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_5.mat')

D = [D1;D2;D3;D4;D5];

count = 0;

k = 10; %Number of codewords

pca_dim = 20;

all_desc = [];

for i=1:1
dataset_desc = dense_sift(D(i));
all_desc = [all_desc;dataset_desc];
end

[signals,pca_dir] = do_pca(all_desc);

signals = signals(1:pca_dim,:);

%cluster
[idx,C] = kmeans(signals',k);

N = 10000; %Number of training samples
[hist] = build_bof(idx,k,N);

signals = signals' ;%Make the rows as the data points


%Load the test data
DTest = load('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/test_batch.mat')

%Extract SIFT descriptors for the test data
test_desc = dense_sift(DTest);

%Project test data along the pca dimensions of the training data
[rowTest,colTest] = size(test_desc);

mnTest = double(mean(test_desc',2));
data = double(test_desc') - repmat(mnTest,1,rowTest);

signalsTest = pca_dir * data;
signalsTest = signalsTest' ;%Make the rows as the data points
signalsTest = signalsTest(:,1:pca_dim);

%Find the closest cluster center (use the clusters found using the training
%samples
idxTest = dsearchn(C,signalsTest);

%Build the bag of features
NTest = 10000;
histTest = build_bof(idxTest,k,NTest);

%Train the k nearest neighbour classifier


%Using this model, on testing data and show errors

