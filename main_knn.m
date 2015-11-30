clear all
clear workspace

DX = load('trainX.mat');
trainX = DX.hist;

DY = load('trainY.mat');
trainY = DY.trainY;

TX = load('testX.mat');
testX = TX.histTest;

TY = load('testY.mat');
testY = TY.testY;

NTest = size(testY,1);

%Classify using n nearest neighbours
n = 20; %Number of nearest neighbours
k = size(testX,2);
predY = perform_knn(trainX,trainY,n,NTest,testX,k);

err = mean(predY ~= testY);

err

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

disp(confMat)