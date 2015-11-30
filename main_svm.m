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

%Classify using SVM
t = templateSVM('KernelFunction','linear');
Mdl = fitcecoc(trainX,trainY,'Learners',t,'FitPosterior',1,...
    'Verbose',2);

predY = predict(Mdl,testX);

err = mean(predY ~= testY);

err

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

disp(confMat)