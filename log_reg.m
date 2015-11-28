D = load('trainX');
trainX = D.hist;

D = load('testX');
testX = D.histTest;

D = load('trainY');
trainY = D.trainY;

D = load('testY');
testY = D.testY;

%Classify using Multinomial Logistic Regression
B = mnrfit(single(trainX),single(trainY+1));

predY = mnrval(B,testX);


err = mean(predY ~= (testY+1));

err;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), predY);

helperDisplayConfusionMatrix(confMat)