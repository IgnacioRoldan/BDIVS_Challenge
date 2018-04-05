function [beta] = sc_pooling(feaSet, V, pyramid, lambda)
%================================================
% 
% Usage:
% Compute the linear spatial pyramid feature using sparse coding. 
%
% Inputss:
% feaSet        -structure defining the feature set of an image   
%                   .feaArr     local feature array extracted from the
%                               image, column-wise
%                   .x          x locations of each local feature, 2nd
%                               dimension of the matrix
%                   .y          y locations of each local feature, 1st
%                               dimension of the matrix
%                   .width      width of the image
%                   .height     height of the image
% V             -sparse dictionary, column-wise
% lambda        -sparsity regularization parameter
% pyramid       -defines structure of pyramid 
% 
% Output:
% beta          -multiscale max pooling feature
%
% Written by Jianchao Yang @ NEC Research Lab America (Cupertino)
% Mentor: Kai Yu
% July 2008
%
% Revised May. 2010
%===============================================

% Vocabulary size
dSize = size(V, 2);
% Number of SIFT descriptors
nSmp = size(feaSet.feaArr, 2);
img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% Initalize sparse codes
sc_codes = zeros(dSize, nSmp);

% Compute sparse codes
beta = 1e-4;
A = V'*V + 2*beta*eye(dSize);
Q = -V'*feaSet.feaArr;
for iter1 = 1:nSmp
    sc_codes(:, iter1) = L1QP_FeatureSign_yang(lambda, A, Q(:, iter1));
end

sc_codes = abs(sc_codes);

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

% For every label in the spatial pyramid
for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    % For every bin the level
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end    
        % Apply max pooling
        beta(:, bId) = max(sc_codes(:, sidxBin), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

% Concatenate and normalize
beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
