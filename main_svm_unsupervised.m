
%% Configuration
addpath minFunc;
rfSize = 6;
numCentroids=160;
whitening=true;
numPatches = 40000;
CIFAR_DIM=[32 32 3];

PROJ_DIR = '/Users/samhitathakur/USC/Projects/EE660/MLProject/';

%% Load CIFAR training data
fprintf('Loading training data...\n');

D = load (strcat(PROJ_DIR,'reduced_data.mat'))

trainX = double(D.ds.data);
trainY = double(D.ds.labels);

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% whiten
if (whitening)
  C = cov(patches);
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
  patches = bsxfun(@minus, patches, M) * P;
end

% run K-means
centroids = run_kmeans(patches, numCentroids, 50);

% extract training features
if (whitening)
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
else
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
end

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

save(strcat(PROJ_DIR,'ustrainX.mat','trainXCs'));
save(strcat(PROJ_DIR,'ustrainY.mat','trainY'));


% train classifier using SVM
C = 100;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

%% Load CIFAR test data
fprintf('Loading test data...\n');
DTest = load('reduced_data_test.mat');

testX = double(DTest.ds_test.data);
testY = double(DTest.ds_test.labels);

% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

save(strcat(PROJ_DIR,'ustestX.mat','testXCs'));
save(strcat(PROJ_DIR,'ustestY.mat','testY'));


% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

% Tabulate the results using a confusion matrix.
confMat = confusionmat(double(testY), double(labels));
confMat = round(100 * confMat/(length(testY)/10));

disp(confMat)

xLabels = [1;2;3;4;5;6;7;8;9;10];
yLabels = xLabels;

heatmap(confMat,xLabels,yLabels,1);

