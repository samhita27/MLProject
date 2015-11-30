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

%Classify using Multinomial Logistic Regression
B = mnrfit(single(trainX),single(trainY+1));

predY = mnrval(B,testX);


err = mean(predY ~= (testY+1));

err

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

helperDisplayConfusionMatrix(confMat)