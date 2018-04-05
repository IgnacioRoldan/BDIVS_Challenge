%% Sparse Coding for Image Classification
% This is an example code for running the ScSPM algorithm described in "Linear 
% Spatial Pyramid Matching using Sparse Coding for Image Classification".
%
% The algorithm is composed of the following parts:
% a) SIFT descriptor extraction.
% b) Sparse coding. Honglak Lee's matlab codes are integrated for
%    training the dictionary.
% c) Multi-scale spatial max pooling of sparse codes.
% d) Linear SVM classification.
%%

close all; clearvars; clc;

%% Libraries
% Add the libraries corresponding to SIFT computation, sparse coding
% computation, and SVM classification.
addpath('libs/sift');
addpath(genpath('libs/sparse_coding'));
addpath('libs/spm_pooling');
addpath('libs/large_scale_svm');
addpath('libs/miscellaneous');


%% Inputs
% The algorithm can be tested on the Caltech101 and handPosesLeapMotion
% datasets. 
%
% The handPosesLeapMotion dataset contains multiple images of 6 different
% hand poses to be recognised. Images have been acquired by the Leap Motion
% sensor that provides near-infrared imagery. The dataset is available in
% the local folder './image/handPosesLeapMotion/'.
%
% Caltech101 dataset contains images of objects belonging to 101
% classes to be classified. The aim is to recognize every object in the
% dataset by predicting the class which it belongs to. The dataset is
% available in the local folder './image/Caltech101'

% Database structures.
databases.name = 'Caltech101';
databases.imageFileExt = 'jpg';
databases.imagePath = './image/Caltech101';
databases(2).name = 'handPosesLeapMotion';
databases(2).imageFileExt = 'png';
databases(2).imagePath = './image/handPosesLeapMotion';
databases(3).name = 'Kaggle';
databases(3).imageFileExt = 'png';
databases(3).imagePath = './image/Kaggle';

% Database selection
inp.database = databases(3); % Selection of handPosesLeapMotion dataset.


%% Parameters
% *System flags*
sys.flags.order_images = true; % Order the images in folder classes
sys.flags.cal_sift = true; % SIFT computation.
sys.flags.dic_training = true; % Dictionary training.
sys.flags.sparseCodes = true; % Sparse coding.
sys.flags.gpu = true; 
sys.flags.Paral = true;

%%
% *SIFT descriptor extraction*
par.sift.gridSpacing = 6;      % Spacing for sampling dense descriptors.
par.sift.patchSize = 16;       % Patch size for extracting SIFT descriptors.
par.sift.maxImSize = 0;      % Maximum size of the input image.  0 value do not resize.
par.sift.nrml_threshold = 1;   % Low contrast region normalization threshold (descriptor length).
par.sift.featurePath = ['./data/', inp.database.name];

%%
% *Sparse coding*
par.sparseCoding.nBases = 1024;      % Number of bases.
par.sparseCoding.nsmp = 20000;       % Number of samples to train the dictionary.
par.sparseCoding.beta = 1e-5;        % Regularization parameter (small for stabilizing sparse coding).
par.sparseCoding.num_iters = 50;     % Number of iterations.
par.sparseCoding.lambda = 0.15;      % Sparsity regularization.
par.sparseCoding.Xpath = ['./dictionary/rand_patches_' inp.database.name '_' num2str(par.sparseCoding.nsmp) '.mat']; % Path to random patches to train the dictionary.
par.sparseCoding.Vpath = ['./dictionary/dict_' inp.database.name '_' num2str(par.sparseCoding.nBases) '.mat']; % Path to pre-trained dictionary V.


%%
% *Multi-scale spatial max pooling*
par.spatialPooling.pyramid = [1, 2, 4]; % Spatial block number on each level of the pyramid.          
par.spatialPooling.knn = 200; % Find the k-nearest neighbors for approximate sparse coding. If set 0, use the standard sparse coding.
par.spatialPooling.SPpath = ['./imageFeatures/imFeat_', inp.database.name, '.mat']; % Path to computed image features.


%%
% *Classification*
par.classif.nRounds = 5;    % Number of random tests.
par.classif.lambda2 = 0.1;  % Regularization parameter for svm classifier.
par.classif.tr_pc = 80;    % Training percentage per category (test percentage is the complementary).


%% Initialization
% Time measurements.
t0 = 0; t1 = 0; t2 = 0; t3 = 0; t4 = 0;
%% 
%
% *Parallel processing* 
if(license('test', 'Distrib_Computing_Toolbox') == false)
    sys.flags.gpu = 0;
    sys.flags.Paral = 0;
    fprintf('Distrib_Computing_Toolbox not available: GPU and multicore capabilities can not be used\n');
end
% GPU device
gpuHandle = [];
if(sys.flags.gpu)
    for ii = 1:gpuDeviceCount
        g = gpuDevice(ii);
        if(str2double(g.ComputeCapability) >= 3)
           gpuHandle = g;
           break; 
        end
    end
end
% Cluster processing
if(sys.flags.Paral)
    p = gcp('nocreate');
    if(isempty(p))
        parpool; % See pool size specified by your parallel preferences and the default profile.
%     parpool('local', 4);
    end
end

%% a0) ORGANISE THE IMAGES

if(sys.flags.order_images)
    disp('==================================================');
    fprintf('Ordering Images...\n');
    disp('==================================================');
    
    t0ini = tic;
    OrderImages(inp.database.imagePath)
    t0=toc(t0ini);
    strTime = seconds2human(t0, 'full');
    fprintf(1,'Time for ordering images: %s\n', strTime);
end

%% a) SIFT DESCRIPTOR EXTRACTION
% The SIFT descriptors are densely sampled from each image on a grid with
% step size 6 pixels and extracted from 16x16 patches. The images are all
% preprocessed into gray scale. A structure is created with the following
% fields:
% - imnum: number of images in the dataset.
% - cname: name of the classes.
% - label: label for every image in the dataset.
% - path: path to the SIFT descriptors for every image.
% - imagePath: path to original images.
% - nclass: number of classes.
%
% SIFT descriptor computation is very intensive, and therefore pre-computed
% SIFT descriptors are also provided. To use them, 'sys.flags.cal_sift'
% must be false.

% Calculate SIFT descriptors or retrieve the feature database directory.
if (sys.flags.cal_sift)
    tic
    % Compute SIFT descriptors. 
    [featDatabase, lenStat] = CalculateSiftDescriptor(inp.database.imagePath, inp.database.imageFileExt, par.sift.featurePath,...
        par.sift.gridSpacing, par.sift.patchSize, par.sift.maxImSize, par.sift.nrml_threshold);
    save([par.sift.featurePath, '/featDatabaseStruct.mat'], 'featDatabase');
    t1=toc;
    strTime = seconds2human(t1, 'full');
    fprintf(1,'Time for SIFT DESCRIPTOR EXTRACTION: %s\n', strTime);
else
    % Retrive paths of pre-computed SIFT descriptors.
    load([par.sift.featurePath, '/featDatabaseStruct.mat']); % Load featDatabase variable.
end


%% b) SPARSE CODING - TRAINING PHASE
% In the training phase, a descriptor set X from a random collection of
% image patches is used to solve Equation 3 with respect to U and V, where
% V is retained as the dictionary. X is composed of approximately 'nsmp'
% random SIFT descriptors. The results of the training are the dictionary V
% (visual words) and descriptors' codes U (for the training samples).
%
% The dictionary computation is intensive, and therefore it is pre-computed
% To use it, set 'sys.flags.dic_training' to false.

% Train the dictionary or load it.
 if (sys.flags.dic_training)
    try
        % If X is alredy computed, load it.
        load(par.sparseCoding.Xpath); % Load X variable.
    catch
        % If not, generate it.
        X = rand_sampling(featDatabase, par.sparseCoding.nsmp);
        save(par.sparseCoding.Xpath, 'X');
    end
    t2ini = tic;
    % Compute V and U. 
    Sigma = eye(par.sparseCoding.nBases);
    [V, U, stat] = reg_sparse_coding(X, par.sparseCoding.nBases, Sigma, par.sparseCoding.beta, ...
        par.sparseCoding.lambda, par.sparseCoding.num_iters);
    t2=toc(t2ini);
    strTime = seconds2human(t2, 'full');
    fprintf(1,'Time for SPARSE CODING - TRAINING PHASE: %s\n', strTime);
    save(par.sparseCoding.Vpath, 'V', 'U', 'stat');
else
    % Load pre-computed V and U.
    load(par.sparseCoding.Vpath); % Load V,U,and stat variables.
end
% Size of the dictionary
nBases = size(V, 2);


%% c) MULTI-SCALE SPATIAL MAX POOLING - CODING PHASE
% In the coding phase, for each image represented as a descriptor set X,
% the sparse codes U are obtained by optimizing Equation 3 with respect to
% U only. The standard sparse coding uses all the visual words in the
% dictionary or vocabulary to approximate X. However, the K-Nearest
% Neighbor technique can be applied to consider only 'knn' sparse codes.
%
% For every image in the dataset, first sparse codes are computed for each
% of its SIFT descriptors. Second, every sparse code is assigned to a bin
% within a specific level in the spatial pyramid. Third, max pooling is
% applied to every bin to get a single feature vector. And finally, all
% the resulting feature vectors are concatenated and normalized to form a
% compact representation (see 'sc_pooling' or 'sc_approx_pooling').

% Computes sparse codes or load them.
if (sys.flags.sparseCodes)
    % Calculate the dimension of the final feature vector for every image.
    dimFea = sum(par.sparseCoding.nBases*par.spatialPooling.pyramid.^2);
    % Number of images.
    numFea = length(featDatabase.path);

    % Initialize matrix containing feature vectors of every image.
    sc_fea = zeros(dimFea, numFea);
    % Initialize matrix containing labels of every image
    sc_label = zeros(numFea, 1);

    disp('==================================================');
    fprintf('Calculating the sparse coding feature...\n');
    fprintf('Regularization parameter: %f\n', par.sparseCoding.lambda);
    disp('==================================================');

    tic
    % For every image in the dataset. 
    for iter1 = 1:numFea  
        if ~mod(iter1, 50)
            fprintf('.\n');
        else
            fprintf('.');
        end

        % Load SIFT descriptors for every image.
        fpath = featDatabase.path{iter1};
        loadedData = load(fpath, 'feaSet'); % Load 'feaSet' variable.

        % Compute sparse codes...
        if (par.spatialPooling.knn)
            % Using K-NN sparse coding.
            sc_fea(:, iter1) = sc_approx_pooling(loadedData.feaSet, V, par.spatialPooling.pyramid, ...
                par.sparseCoding.lambda, par.spatialPooling.knn);
        else
            % Using the standard sparse coding.
            sc_fea(:, iter1) = sc_pooling(loadedData.feaSet, V, par.spatialPooling.pyramid, par.sparseCoding.lambda);
        end
        sc_label(iter1) = featDatabase.label(iter1);
    end
    save(par.spatialPooling.SPpath, 'sc_fea', 'sc_label');
    t3=toc;   
    strTime = seconds2human(t3, 'full');
    fprintf(1,'\nTime for MULTI-SCALE SPATIAL MAX POOLING - CODING PHASE: %s\n', strTime);
else
    load(par.spatialPooling.SPpath); % Load 'sc_fea' and 'sc_label' variables.
end



%% d) CLASSIFICATION
% The performance of this algorithm for the computed features can be
% evaluated by a linear SVM. The performance is measured in terms of
% accuracy (number of correctly recognized objects/total number of
% objects).
%
% Every time we evaluate the classification accuracy, we are using a
% different set of training samples, so that the result will be different
% in each iteration (stochastic process). Therefore, we repeat the
% experimental process by 'par.classif.nRounds' times with different random
% selected training and testing images to obtain reliable results. The
% average per class recognition rates are recorded for each run, and we
% report the final results by the mean and standard deviation of the
% recognition rates.

% Extract feature dimension, number of features, labels, and number of
% classes.
[dimFea, numFea] = size(sc_fea);
clabel = unique(sc_label);
nclass = length(clabel);

% Initialize the accuracy.
accuracy = zeros(par.classif.nRounds, 1);

tic
% For every round (experiment).
for ii = 1:par.classif.nRounds
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    % For every class, selection of training and testing samples.
    for jj = 1:nclass
        idx_label = find(sc_label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        tr_num = round(num*par.classif.tr_pc/100);
        if(tr_num == num)
            error('Not enough test data for class: %s', clabel(jj));
        end
        
        % Select random samples for training and testing.
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    % Training samples
    tr_fea = sc_fea(:, tr_idx);
    tr_label = sc_label(tr_idx);
    
    % Test samples
    ts_fea = sc_fea(:, ts_idx);
    ts_label = sc_label(ts_idx);
    
    trainSVMTime = tic;
    % Training SVM.
    [w, b, class_name] = li2nsvm_multiclass_lbfgs(tr_fea', tr_label, par.classif.lambda2);
    ttr = toc(trainSVMTime); 
    strTime = seconds2human(ttr, 'full');
    fprintf(1,'Time for Training SVM: %s\n', strTime);

    predSVMTime = tic;
    % Prediction.
    [C, Y] = li2nsvm_multiclass_fwd(ts_fea', w, b, class_name);
    tpr = toc(predSVMTime);
    strTime = seconds2human(tpr, 'full');
    fprintf(1,'Time for Prediction SVM: %s\n', strTime);
    
    % Show some prediction samples.
    figure(1);
    set(gcf, 'Position', get(0,'Screensize'))
    p = randperm(length(ts_idx));
    for i = 1:16
        subplot(4,4,i);
        I = imread(featDatabase.imagePath{ts_idx(p(i))});
        text = sprintf('P:%s,Gt:%s', featDatabase.cname{C(p(i))}, featDatabase.cname{ts_label(p(i))});
        I = insertText(I, [1,1], text, 'FontSize', 36, 'TextColor' , 'red');
        imshow(I);
        drawnow;
    end
    pause(3);
   
    % Compute accuracy.
    acc = zeros(length(class_name), 1);
    for jj = 1 : length(class_name)
        c = class_name(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end
    
    accuracy(ii) = mean(acc); 
end

t4=toc;  
strTime = seconds2human(t4, 'full');
fprintf(1,'Time for CLASSIFICATION: %s\n', strTime);
strTime = seconds2human(t1+t2+t3+t4, 'full');
fprintf(1,'Total time: %s\n', strTime);

% Print results
fprintf('Mean accuracy: %f\n', mean(accuracy));
fprintf('Standard deviation: %f\n', std(accuracy));