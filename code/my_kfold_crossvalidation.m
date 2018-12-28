% Soroosh Shalileh Master Thesis "Crowd Scenes Analysis"
% Crowd Scene Analysis using Sliding window bbox detection with linear SVM
% 5 fold Cross Validation data prepration for each classes

%% Abnormal Peds:
% addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\SlidingWin_Trckr\data\Positives_aped_modified')
% load('UCSD1_Aped.mat')
%**************************************************************************
% addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\SlidingWin_Trckr\data\Positive_biker_modified')
% load('UCSD1_Biker.mat')
%**************************************************************************
% addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid\data\Positives_skater_modified')
% load('UCSD1_Skater.mat')
%**************************************************************************
addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\sf_hog_svm_slid\data\Positives_cart_modified')
load('UCSD1_Cart.mat')
num_images = size(UCSD1_Cart.TotalImages2read,1) ; 


% k fold data prepration :
for k=1:5
    
    [Train(:,k), Test(:,k)] =  crossvalind('HoldOut',num_images,0.3) ;
    
    trainImages(:,k)   =   UCSD1_Cart.TotalImages2read(Train(:,k)) ;
    
    testImages(:,k)    =   UCSD1_Cart.TotalImages2read(Test(:,k)) ;

end


Kfold_Cart.train = trainImages ; 
Kfold_Cart.test  = testImages ;
