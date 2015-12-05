function predY = perform_knn(trainX,trainY,n,NTest,testX,k)
%Train the k nearest neighbour classifier
mdl = fitcknn(trainX,trainY,'NumNeighbors',n);

%Using this model, on testing data and show errors
predY = zeros(NTest,1);
for i=1:NTest
    predY(i,1) = predict(mdl,testX(i,1:k));
end

end