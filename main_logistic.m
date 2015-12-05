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
B = mnrfit(single(trainX),single(trainY));

probY = mnrval(B,testX);

predY = zeros(NTest,1);

for i=1:NTest
   [num] = max(probY(i,:));
   [x y] = ind2sub(size(probY(i,:)),find(probY(i,:)==num))
   predY(i,1) = y;
end



err = mean(predY ~= (testY));

err

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);