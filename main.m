clear all
clear workspace

D1 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_1.mat')
[row,col] = size(D1.data)

all_descriptors = [];
count = 0;


for i= 1:row
    R=D1.data(i,1:1024);
    G=D1.data(i,1025:2048);
    B=D1.data(i,2049:3072);
    A(:,:,1)=reshape(R,32,32);
    A(:,:,2)=reshape(G,32,32);
    A(:,:,3)=reshape(B,32,32);
    I = single(rgb2gray(A));
    [f,d] = vl_sift(I);
    if isempty(d)
        count = count + 1;
        disp(i);
    end
    all_descriptors = [all_descriptors;d'];
end

disp(count);