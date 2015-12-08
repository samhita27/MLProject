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
n = [100 200 300]; %Number of nearest neighbours
d = size(testX,2); %Dimensions of feature vector

nfold = 10;

kloss = zeros(size(n,2),1);

% Find the cross validation errors for different values of n and find best
% n
for i=1:size(n,2)
    mdl = fitcknn(trainX,trainY,'NumNeighbors',n(i));
    cvmdl = crossval(mdl,'KFold',nfold);
    kloss(i,1) = kfoldLoss(cvmdl)
end

[val,ind] = min(kloss);

fprintf('Performing knn using n=%d\n',n(ind));
predY = perform_knn(trainX,trainY,n(ind),NTest,testX,d);

err = mean(predY ~= testY);

fprintf('Test samples classification accuracy %f%%',(1-err)*100);
fprintf('\n');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

confMat = confusionmat(double(testY), double(predY));
confMat = round(100 * confMat/(length(testY)/10));

disp(confMat)

xLabels = [1;2;3;4;5;6;7;8;9;10];
yLabels = xLabels;

heatmap(confMat,xLabels,yLabels,1);

