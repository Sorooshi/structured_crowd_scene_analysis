% Soroosh Shalileh Master Thesis - Crowd Scens Analysis HoG Sliding Window
% Crowd Scene Analysis using Sliding window bbox detection with linear SVM
% Extracting cropped images, and Grount Truth for each targer class
close all
clear
%% Step 0. Loading data
addpath('C:\Users\Soroosh\Documents\MATLAB\SorooshThesis\code\SlidingWin_Trckr\data')
load('ROIs_Lables_test_only');
train_data = ROIs_Lables_test_only;
%Parameter:
feature_params = struct('bbox_size', 35,'template_size',36, ...
    'hog_cell_size', 6); % template size use for negative feature
%Determing categories:
classes = ROIs_Lables_test_only.Properties.VariableNames(3:end);
num_classes = size(classes,2);
% classes = {'Normal','Bicycler','Skater','Cart','Abnormal Peds'};
% image cropp dimension :
width = 19; % weight
height = 31; % height
%% Step 1. croping the images 
train_data_pos = table2struct(ROIs_Lables_test_only);
disp('start to croping the ROIs')

[UCSD1_Skater] = img_crop(train_data_pos,width,height);
    disp('croption is finished')


