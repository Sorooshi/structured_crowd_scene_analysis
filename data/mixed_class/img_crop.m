% Soroosh Shalileh Master Thesis
% implementations by Soroosh Shalileh
% Extracting cropped images, and Grount Truth for each targer class

function [UCSD1]=...
    img_crop(train_data_pos,width,height)
%% prelimanary
% Number of images for training
images_num = size(train_data_pos,1);
%% feature extraction
disp ('Hog Feature Extraction')
UCSD1 = {};
for i=1:images_num
    disp(i)  %for debuge
    img = im2single(imread(train_data_pos(i).imageFilename));
    
    if ~isempty(train_data_pos(i).Bicycler(:))
        
        num_bbox = size(train_data_pos(i).Bicycler(1:end,:),1);
        
        for j=1:num_bbox
            bbox_biker(j,:) = [train_data_pos(i).Bicycler(j,1)-2,...
                train_data_pos(i).Bicycler(j,2)-3,width,height ];
            
%             img_crop = imcrop(img,bbox(j,:));
%             img_name = sprintf('Bicycler_i%d_j%d.jpg', i,j) ;
%             imwrite(img_crop,img_name,'jpg')
            
            trainImages_biker{i,j} = train_data_pos(i).imageFilename;
            trainBoxes_biker{i,j} = {[bbox_biker(j,1), bbox_biker(j,2),...
                (bbox_biker(j,1)+width),(bbox_biker(j,2)+height)]};
            trainBoxImages_biker(:,i,j) = [bbox_biker(j,1), bbox_biker(j,2),...
                (bbox_biker(j,1)+width),(bbox_biker(j,2)+height)];
            trainBoxLabels_biker{i,j} = 'Bicycler';
            
        end
    end
    
    if ~isempty(train_data_pos(i).Skater(:))
        
        num_bbox = size(train_data_pos(i).Skater(1:end,:),1);
        
        for j=1:num_bbox
            bbox_skater(j,:) = [train_data_pos(i).Skater(j,1)-2,...
                train_data_pos(i).Skater(j,2)-3,width,height ];
            
%             img_crop = imcrop(img,bbox(j,:));
%             img_name = sprintf('Skater_i%d_j%d.jpg', i,j) ;
%             imwrite(img_crop,img_name,'jpg')
            
            trainImages_skater{i,j} = train_data_pos(i).imageFilename;
            trainBoxes_skater{i,j} = {[bbox_skater(j,1), bbox_skater(j,2),...
                (bbox_skater(j,1)+width),(bbox_skater(j,2)+height)]};
            trainBoxImages_skater(:,i,j) = [bbox_skater(j,1), bbox_skater(j,2),...
                (bbox_skater(j,1)+width),(bbox_skater(j,2)+height)];
            trainBoxLabels_skater{i,j} = 'Skater';
            
            
        end
    end
    
    if ~isempty(train_data_pos(i).Abnormal_Peds(:))
        
        num_bbox = size(train_data_pos(i).Abnormal_Peds(1:end,:),1);
        
        for j=1:num_bbox
            bbox_Aped(j,:) = [train_data_pos(i).Abnormal_Peds(j,1)-2,...
                train_data_pos(i).Abnormal_Peds(j,2)-3,width,height ];
            
%             img_crop = imcrop(img,bbox(j,:));
%             img_name = sprintf('Aped_Peds_i%d_j%d.jpg', i,j) ;
%             imwrite(img_crop,img_name,'jpg')
%             
            trainImages_aped{i,j} = train_data_pos(i).imageFilename;
            trainBoxes_aped{i,j} = {[bbox_Aped(j,1), bbox_Aped(j,2),...
                (bbox_Aped(j,1)+width),(bbox_Aped(j,2)+height)]};
            trainBoxImages_aped(:,i,j) = [bbox_Aped(j,1), bbox_Aped(j,2),...
                (bbox_Aped(j,1)+width),(bbox_Aped(j,2)+height)];
            trainBoxLabels_aped{i,j} = 'Aped';
            
            
        end
    end
    
end
%****************

UCSD1.Images.Biker  = trainImages_biker    (~cellfun('isempty',trainImages_biker));
UCSD1.Images.Skater = trainImages_skater     (~cellfun('isempty',trainImages_skater));
UCSD1.Images.Aped   = trainImages_aped    (~cellfun('isempty',trainImages_aped));
UCSD1.Images.Total = [UCSD1.Images.Biker; UCSD1.Images.Skater; UCSD1.Images.Aped];

% UCSD1.Boxes.Biker   = trainBoxes_biker   (~cellfun('isempty',trainBoxes_biker));
% UCSD1.Boxes.Skater  = trainBoxes_skater (~cellfun('isempty',trainBoxes_skater));
% UCSD1.Boxes.Aped    = trainBoxes_skater   (~cellfun('isempty',trainBoxes_aped));
% UCSD1.Boxes.Total = [UCSD1.Boxes.Biker; UCSD1.Boxes.Skater; UCSD1.Boxes.Aped] ;


UCSD1.BoxImages.Biker   = trainBoxImages_biker(:,any(trainBoxImages_biker,1));
UCSD1.BoxImages.Skater   = trainBoxImages_skater(:,any(trainBoxImages_skater,1));
UCSD1.BoxImages.Aped   = trainBoxImages_aped(:,any(trainBoxImages_aped,1));
UCSD1.BoxImages.Total = [UCSD1.BoxImages.Biker, UCSD1.BoxImages.Skater, UCSD1.BoxImages.Aped] ;

UCSD1.TotalImages2read = unique(UCSD1.Images.Total);
%****************

    
end

% index = find(strcmp(UCSD1.Images.Total{i},UCSD1.Images.Total))
% UCSD1.BoxImages.Total(:,index)
% UCSD1.TotalImages2read = unique(UCSD1.Images.Total);
