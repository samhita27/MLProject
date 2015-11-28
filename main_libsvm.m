clear all
clear workspace

VLFEATROOT = '/Users/samhitathakur/USC/Projects/EE660/vlfeat-0.9.20'

% addpath to the libsvm toolbox
addpath('/Users/samhitathakur/USC/Projects/EE660/libsvm-3.20/matlab');

s = strcat(VLFEATROOT,'/toolbox/vl_setup');

run(s);

DTrain = load('dSiftTraining');
all_desc = DTrain.all_desc;

DTest = load('dSiftTest');
test_desc = DTest.test_desc;

DTrain = load('trainY');
trainY = DTrain.trainY;

DTest = load('testY');
testY = DTest.testY;

k = 100; %Number of codewords

num_of_datasets = 5

pca_dim = 20;

NTrain = num_of_datasets * 10000; %Number of training samples

[signals,pca_dir] = do_pca(all_desc);

signals = signals(1:pca_dim,:);

%cluster
[idx,C] = kmeans(signals',k);

[hist] = build_bof(idx,k,NTrain);

signals = signals' ;%Make the rows as the data points


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

%Classify using SVM
t = templateSVM('KernelFunction','linear');
Mdl = fitcecoc(hist,trainY,'Learners',t,'FitPosterior',1,...
    'Verbose',2);

predY = predict(Mdl,histTest);

err = mean(predY ~= testY);

err;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

disp(confMat)