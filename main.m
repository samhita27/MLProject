clear all
clear workspace

VLFEATROOT = '/Users/samhitathakur/USC/Projects/EE660/vlfeat-0.9.20'

s = strcat(VLFEATROOT,'/toolbox/vl_setup');

run(s);

D1 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_1.mat')
[row,col] = size(D1.data)

all_descriptors = [];
count = 0;

k = 50; %Number of codewords

d = []; % #D array to store descriptors for each image

count_descr = zeros(row,1) ; %Number of descriptors for each image

for i= 1:row
    R=D1.data(i,1:1024);
    G=D1.data(i,1025:2048);
    B=D1.data(i,2049:3072);
    A(:,:,1)=reshape(R,32,32);
    A(:,:,2)=reshape(G,32,32);
    A(:,:,3)=reshape(B,32,32);
    I = single(rgb2gray(A));
    [frame,descr] = vl_phow(I,'Verbose',2,'Sizes',7,'Step',5,'Color','gray');
    d(:,:,i) = descr';
    count_descr(i,1) = size(descr,2);%Number of columns of descr gives the num of feature vectors for the image
    all_descriptors = [all_descriptors;descr'];
end

[dsift_pca_descr,score,latent] = pca(double(all_descriptors'));

[idx,C] = kmeans(double(dsift_pca_descr),k);

bow = zeros(row,k);

t = 1;

for i=1:row
   for j = t:t+count_descr(i,1)-1
       bow(i,idx(j))= bow(i,idx(j)) + 1;
   end
   t = t + count_descr(i,1);
end




