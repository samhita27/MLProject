
addpath minFunc ;
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

% train classifier using SVM
C = 100;
theta = train_svm(trainX, trainY, C);

[val,labels] = max(trainX*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

% test and print result
[val,labels] = max(testX*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));


% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), double(labels));
confMat = round(100 * confMat/(length(testY)/10));

disp(confMat)

xLabels = [1;2;3;4;5;6;7;8;9;10];
yLabels = xLabels;

heatmap(confMat,xLabels,yLabels,1);

 