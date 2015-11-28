clear all
clear workspace

D1 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_1.mat')
D2 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_2.mat')
D3 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_3.mat')
D4 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_4.mat')
D5 = load ('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/data_batch_5.mat')

D = [D1;D2;D3;D4;D5];

orig_rows_train = 50000;
row_per_dataset = 10000;
num_categories = 10;
rows_per_class = orig_rows_train/num_categories;
num_datasets = 5;


%User defined
num_per_class = 100; %How many samples per class

cat = zeros(rows_per_class,32*32*3,10);

%Split the data into classes
for p=1:num_categories
    k=1;
    for j=1:num_datasets
        for i=1:row_per_dataset
            if(D(j).labels(i,1) == p)
                cat(k,:,p) = D(j).data(i,:);
                k = k+1;
            end
        end
    end
end
    
%Randomly pick a specified number of data from each class
% data = zeros(num_per_class*num_categories,32*32*3);
data = [];
labels = [];
rng('default');

for i=1:num_categories
    rng(i);
    Ind = randperm(rows_per_class,num_per_class);
    temp = cat(Ind,:,i);
    data = [data;temp];
    l = ones(num_per_class,1)*i;
    labels = [labels;l];
end

data = horzcat(data,labels);
%shuffle the rows
data =  data(randperm(num_per_class*num_categories,num_per_class*num_categories),:);

cat_label = data(:,3073);
data(:,3073) = [];






