 function [dataset_descriptors] = dense_sift(S)
 [row,col] = size(S.data)
 
 all_descriptors = [];
 
 for i= 1:row
    R=S.data(i,1:1024);
    G=S.data(i,1025:2048);
    B=S.data(i,2049:3072);
    A(:,:,1)=reshape(R,32,32);
    A(:,:,2)=reshape(G,32,32);
    A(:,:,3)=reshape(B,32,32);
    I = single(rgb2gray(A));
    [frame,descr] = vl_phow(I,'Verbose', 2, 'Sizes', 7, 'Step', 5,'Color','gray');
    all_descriptors = [all_descriptors;descr'];
 end

 dataset_descriptors = all_descriptors;
 

 
 end