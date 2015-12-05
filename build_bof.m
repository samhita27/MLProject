function [hist] = build_bof(idx,k,N)


%Closest centroid
hist = zeros(N,k);

t=1;
for i=1:N
    for j = 1:62 %Replace with start to end for the descriptor
    hist(i,idx(t))= hist(i,idx(t)) + 1; 
    t=t+1;
    end
end

end