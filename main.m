clear all
clear workspace

VLFEATROOT = '/Users/samhitathakur/USC/Projects/EE660/vlfeat-0.9.20'

s = strcat(VLFEATROOT,'/toolbox/vl_setup');

run(s);

D1 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_1.mat')
D2 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_2.mat')
D3 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_3.mat')
D4 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_4.mat')
D5 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_5.mat')

D = [D1;D2;D3;D4;D5];

count = 0;

k = 10; %Number of codewords

pca_dim = 20;

% count_descr1 = zeros(row,1) ; %Number of descriptors for each image

all_desc = [];

for i=1:1
dataset_desc = dense_sift(D(i));
all_desc = [all_desc;dataset_desc];
end

[signals,pca_dir] = do_pca(all_desc);

signals = signals(1:pca_dim,:);

[hist,C] = build_bof(signals);

signals = signals' ;%Make the rows as the data points



