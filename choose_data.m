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

cat_tr = zeros(rows_per_class,32*32*3,10);

%Split the data into classes
for p=1:num_categories
    k=1;
    for j=1:num_datasets
        for i=1:row_per_dataset
            if(D(j).labels(i,1) == p)
                cat_tr(k,:,p) = D(j).data(i,:);
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
    temp = cat_tr(Ind,:,i);
    data = [data;temp];
    l = ones(num_per_class,1)*i;
    labels = [labels;l];
end

data = horzcat(data,labels);
%shuffle the rows
data =  data(randperm(num_per_class*num_categories,num_per_class*num_categories),:);

cat_label = data(:,3073);
data(:,3073) = [];

ds = struct('data',data,'labels',cat_label);

save('reduced_data.mat','ds');

%Choose test data
orig_rows_test = 10000;
samples_per_class = 50;
rows_test_per_class = orig_rows_test/samples_per_class;

T = load('/Users/samhitathakur/USC/Projects/EE660/cifar-10-batches-mat/test_batch.mat');

cat_test = zeros(rows_test_per_class,32*32*3,10);

%Split the data into classes
for p=1:num_categories
    k=1;
        for i=1:orig_rows_test
            if(T.labels(i,1) == p)
                cat_test(k,:,p) = T.data(i,:);
                k = k+1;
            end
        end
end

%Randomly pick a specified number of data from each class
data_test = [];
labels_test = [];
rng('default');

for i=1:num_categories
    rng(i);
    Ind = randperm(rows_test_per_class,samples_per_class);
    temp = cat_test(Ind,:,i);
    data_test = [data_test;temp];
    l = ones(samples_per_class,1)*i;
    labels_test = [labels_test;l];
end

data_test = horzcat(data_test,labels_test);
%shuffle the rows
data_test =  data_test(randperm(samples_per_class*num_categories,samples_per_class*num_categories),:);

cat_label_test = data_test(:,3073);
data_test(:,3073) = [];

ds_test = struct('data',data_test,'labels',cat_label_test);

save('reduced_data_test.mat','ds_test');


