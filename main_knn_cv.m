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
n = [10 15 20 25 26 28 29 30]; %Number of nearest neighbours
d = size(testX,2); %Dimensions of feature vector

nfold = 10;

kloss = zeros(size(n,2),1);

% Find the cross validation errors for different values of n and find best
% n
for i=1:size(n,2)
    mdl = fitcknn(trainX,trainY,'NumNeighbors',n(i));
    cvmdl = crossval(mdl);
    kloss(i,1) = kfoldLoss(cvmdl)
end

[val,ind] = min(kloss);

predY = perform_knn(trainX,trainY,n(ind),NTest,testX,d);

err = mean(predY ~= testY);

err

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

disp(confMat)

