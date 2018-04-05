function [X] = rand_sampling(training, num_smp)
% Sample local features for unsupervised codebook training

% Number of images
num_img = length(training.label);
% Number of features per image
num_per_img = round(num_smp/num_img);
% Number of features in X to train the dictionary
num_smp = num_per_img*num_img;

% Compute the dimension of features
load(training.path{1});
dimFea = size(feaSet.feaArr, 1);

% Generate the set of training features X
X = zeros(dimFea, num_smp);

cnt = 0;
for ii = 1:num_img
    fpath = training.path{ii};
    load(fpath);
    num_fea = size(feaSet.feaArr, 2);
    rndidx = randperm(num_fea);
    X(:, cnt+1:cnt+num_per_img) = feaSet.feaArr(:, rndidx(1:num_per_img));
    cnt = cnt+num_per_img;
end
