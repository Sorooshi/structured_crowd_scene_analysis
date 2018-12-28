% Soroosh Shalileh Master Thesis "Crowd Scenes Analysis" 

% Abnormality Detection and Recognition in Crowd Scenes 
% Based on structured Output Regression and Histogram of Oriented Gradient
% utilizing SVM and Hard Negative mining 

% Based on the code which have been published in:
% https://github.com/vedaldi/practical-object-category-detection

setup ;
% Training cofiguration
targetClass = 1 ;   

numHardNegativeMiningIterations = 7 ;
% iterationSchedule = [ 200 200 400 400 800 1361 1361 ];  % Biker
% iterationSchedule = [50 50 100 350 350 460 460] ;       % Aped
iterationSchedule = [50 50 100 350 350 350 428 ] ;      % Cart 428 428
% iterationSchedule = [100 100 200 350 350 350 774 774 774 ] ;       % Skater 350 200 774 
% Scale space configuration
hogCellSize = 12 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
    minScale,...
    maxScale,...
    numOctaveSubdivisions*(maxScale-minScale+1)) ;

%% Load data
trainImages = {} ;
trainBoxes = [] ;
trainBoxPatches = {} ;
trainBoxImages = {} ;
trainBoxLabels = [] ;

% Compute HOG features of examples
trainBoxHog = {} ;
% Biker :
% names = dir('data/Positive_biker_modified/*.jpg') ;
% names = fullfile('data', 'Positive_biker_modified', {names.name}) ;
%**************************************************************************
% Cart :
names = dir('data/Positives_cart_modified/*.jpg') ;
names = fullfile('data', 'Positives_cart_modified', {names.name}) ;
%**************************************************************************
% Abnormal Peds :
% % % names = dir('data/Positives_aped_modified/*.jpg') ;
% % % names = fullfile('data', 'Positives_aped_modified', {names.name}) ;
%**************************************************************************
% Skater :
% % % % names = dir('data/Positives_skater_modified/*.jpg') ;
% % % % names = fullfile('data', 'Positives_skater_modified', {names.name}) ;
%**************************************************************************
% Non Cart Abnormality :
% % % % % names = dir('data/mixed_class/*.jpg') ;
% % % % % names = fullfile('data', 'mixed_class', {names.name}) ;

for i=1:numel(names)
    im = imread(names{i}) ;
    im = imresize(im, [64 64]) ;
    %   trainBoxes(:,i) = [0.5 ; 0.5 ; 64.5 ; 64.5] ;
    trainBoxPatches{i} = im2single(im) ;
    %   trainBoxImages{i} = names{i} ;
    trainBoxLabels(i) = 1 ;
end

trainBoxPatches = cat(4, trainBoxPatches{:}) ;

% Compute HOG features of examples 
trainBoxHog = {} ;
for i = 1:size(trainBoxPatches,4)
    trainBoxHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainBoxHog = cat(4, trainBoxHog{:}) ;
modelWidth = size(trainBoxHog,2) ;
modelHeight = size(trainBoxHog,1) ;

% -------------------------------------------------------------------------
% Train with hard negative mining for each class
% -------------------------------------------------------------------------

% Initial positive and negative data
pos = trainBoxHog(:,:,:,ismember(trainBoxLabels,targetClass)) ;
neg = zeros(size(pos,1),size(pos,2),size(pos,3),0) ;
% -------------------------------------------------------------------------
% Loading train data :
% Biker Data : 
% addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid_4Presentation\data\Positive_biker_modified')
% load('UCSD1_Biker.mat') 
% load('Kfold_Biker.mat') 
% trainImages     = Kfold_Biker.train(:,5)  ;
% trainBoxes      = UCSD1_Biker.BoxImages ;
% trainBoxImages  = UCSD1_Biker.Images ;
%**************************************************************************
% Cart Data :
addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid_4Presentation\data\Positives_cart_modified')
load('UCSD1_Cart.mat')  
load('Kfold_Cart.mat') 
trainImages     = Kfold_Cart.train(:,2)  ;
trainBoxes      = UCSD1_Cart.BoxImages ;
trainBoxImages  = UCSD1_Cart.Images  ;
%**************************************************************************
% Abnormal Peds Data :
% % % addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid_4Presentation\data\Positives_aped_modified')
% % % load('UCSD1_Aped.mat')
% % % load('Kfold_Aped.mat')
% % % trainImages     = Kfold_Aped.train(:,5) ;
% % % trainBoxes      = UCSD1_Aped.BoxImages ;
% % % trainBoxImages  = UCSD1_Aped.Images ;
%**************************************************************************
% Skater Data :
% % % % addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid_4Presentation\data\Positives_skater_modified')
% % % % load('UCSD1_Skater.mat')
% % % % load('Kfold_Skater.mat') 
% % % % trainImages     = Kfold_Skater.train(:,5)  ;
% % % % trainBoxes      = UCSD1_Skater.BoxImages ;
% % % % trainBoxImages  = UCSD1_Skater.Images ;
%**************************************************************************
% Non Cart Abnormality Data :
% % % % % addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid_4Presentation\data\data\mixed_class')
% % % % % load('UCSD1_NonCart.mat')
% % % % % trainImages     = UCSD1_NonCart.TotalImages2read ;
% % % % % trainBoxes      = UCSD1_NonCart.BoxImages.Total ;
% % % % % trainBoxImages  = UCSD1_NonCart.Images.Total ;

%% Hard Mining and training step :
for t=1:numHardNegativeMiningIterations
    numPos = size(pos,4) ;
    numNeg = size(neg,4) ;
    C = 1 ;
    lambda = 1 / (C * (numPos + numNeg)) ;
    
    fprintf('Hard negative mining iteration %d: pos %d, neg %d\n', ...
        t, numPos, numNeg) ;
    
    x = cat(4, pos, neg) ;
    x = reshape(x, [], numPos + numNeg) ;
    y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;
    w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;  %0.01
    w = single(reshape(w, modelHeight, modelWidth, [])) ;
    
    % Plot model
    figure(1) ; clf ;
    imagesc(vl_hog('render', w)) ;
    colormap gray ; axis equal ;
    title(sprintf('SVM HOG model (retraining ieration %d)',t)) ;
    
    % Evaluate on training data and mine hard negatives
    figure(2) ; set(gcf, 'name', sprintf('Retraining iteration %d',t)) ;
    [matches, moreNeg] = ...
        evaluate_model(...
        vl_colsubset(trainImages', iterationSchedule(t), 'beginning'), ...
        trainBoxes, trainBoxImages, ...
        w, hogCellSize, scales ) ; %hard_minning
    
    % Add negatives
    neg = cat(4, neg, moreNeg) ;
    
    % Remove negative duplicates
    z = reshape(neg, [], size(neg,4)) ;
    [~,keep] = unique(z','stable','rows') ;
    neg = neg(:,:,:,keep) ;
end

%% Evaluation Step : (on test scenes)
% Biker : k 1 to 5
% testImages     = Kfold_Biker.test(:,5) ;
% testBoxes      = UCSD1_Biker.BoxImages  ;
% testBoxImages  = UCSD1_Biker.Images ;
% [matches_Biker_k5] = evaluate_model(testImages, ...
%     testBoxes, testBoxImages , ...
%     w, hogCellSize, scales) ;
%**************************************************************************
% % % Cart : k 1 to 5
testImages     = Kfold_Cart.test(:,2)  ;
testBoxes      = UCSD1_Cart.BoxImages ;
testBoxImages  = UCSD1_Cart.Images;
[matches_Cart_k2] = evaluate_model(testImages, ...
    testBoxes, testBoxImages , ...  
    w, hogCellSize, scales) ;
%**************************************************************************
% Abnormal Peds : k = 1 to 5
% % % testImages     = Kfold_Aped.test(:,5);
% % % testBoxes      = UCSD1_Aped.BoxImages ;
% % % testBoxImages  = UCSD1_Aped.Images ;
% % % [matches_Aped_k5] = evaluate_model(testImages, ...
% % %     testBoxes, testBoxImages , ...  
% % %     w, hogCellSize, scales) ;
%**************************************************************************
% Skater : k 1 to 5
% % % % testImages     = Kfold_Skater.test(:,5) ;
% % % % testBoxes      = UCSD1_Skater.BoxImages ;
% % % % testBoxImages  = UCSD1_Skater.Images ;
% % % % [matches_Skater_k5] = evaluate_model(testImages, ...
% % % %     testBoxes, testBoxImages , ...  
% % % %     w, hogCellSize, scales) ;
%**************************************************************************
% Non Cart Abnormality Detection :
% testImages     = UCSD1_NonCart.TotalImages2read ;
% testBoxes      = UCSD1_NonCart.BoxImages.Total ;
% testBoxImages  = UCSD1_NonCart.Images.Total ;
% [matches_noCart_without_tracker] = evaluate_model(testImages, ...
%     testBoxes, testBoxImages , ...  
%     w, hogCellSize, scales) ;
%**************************************************************************
% % figure; vl_pr([test_1.labels], [test_1.scores])
% % figure; vl_pr([test_2.labels], [test_2.scores])
% % figure; vl_pr([test.labels], [test.scores])