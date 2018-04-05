function [V, U, stat] = reg_sparse_coding(X, num_bases, Sigma, beta, lambda, num_iters, batch_size, initV, fname_save)
%
% Regularized sparse coding
%
% Inputs
%       X           -data samples, column wise
%       num_bases   -number of bases
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       lambda      -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%       initV       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       V           -learned dictionary
%       U           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = lambda;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

if ~isa(X, 'double'),
    X = cast(X, 'double');
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('Results/reg_sc_b%d_%s', num_bases, datestr(now, 30));	
end;

% pars

% initialize basis
if ~exist('initV') || isempty(initV)
    V = rand(pars.patch_size, pars.num_bases)-0.5;
	V = V - repmat(mean(V,1), size(V,1),1);
    V = V*diag(1./sqrt(sum(V.*V)));
else
    disp('Using initial V...');
    V = initV;
end

[L M]=size(V);

t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

% optimization loop
while t < pars.num_trials
    t=t+1;
    start_time= cputime;
    stat.fobj_total=0;    
    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
    
    sparsity = [];
    
    for batch=1:(size(X,2)/pars.batch_size),
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx);
        
        % learn coefficients (conjugate gradient)   
        U = L1QP_FeatureSign_Set(Xb, V, Sigma, pars.beta, pars.gamma);
        
        sparsity(end+1) = length(find(U(:) ~= 0))/length(U(:));
        
        % get objective
        [fobj] = getObjective_RegSc(Xb, V, U, Sigma, pars.beta, pars.gamma);       
        stat.fobj_total = stat.fobj_total + fobj;
        % update basis
        V = l2ls_learn_basis_dual(Xb, U, pars.VAR_basis);
    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    
    fprintf(['epoch= %d, sparsity = %f, fobj= %f, took %0.2f ' ...
             'seconds\n'], t, mean(sparsity), stat.fobj_avg(t), stat.elapsed_time(t));
         
    % save results
    fprintf('saving results ...\n');
    experiment = [];
    experiment.matfname = sprintf('%s.mat', pars.filename);     
    save(experiment.matfname, 't', 'pars', 'V', 'stat');
    fprintf('saved as %s\n', experiment.matfname);
end

return

%% 

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
