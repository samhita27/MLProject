function [hist,C] = build_bof(signals,k)

[idx,C] = kmeans(signals',k);

%Closest centroid
hist = zeros(50000,k);

t=1;
for i=1:50000
    for j = 1:9 %Replace with start to end for the descriptor
    hist(i,idx(t))= hist(i,idx(t)) + 1; 
    t=t+1;
    end
end

end