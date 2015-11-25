function [signals,pca_dir] = do_pca(all_desc)

pca_dir = pca(double(all_desc));

pca_dir = pca_dir'; %Now Rows are the PCA components 

[row,col] = size(all_desc);

mn = double(mean(all_desc',2));
data = double(all_desc') - repmat(mn,1,row);

signals = pca_dir * data;

end
