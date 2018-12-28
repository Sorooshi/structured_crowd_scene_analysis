% Soroosh Shalileh Master Thesis - Crowd Scens Analysis HoG Sliding Window
% implementations by Soroosh Shalileh
% Extracting cropped images, and Grount Truth for each targer class

function [UCSD1] = img_crop(train_data_pos,width,height)

images_num = size(train_data_pos,1);

disp ('Hog Feature Extraction')
UCSD1 = {};
for i=1:images_num
    disp(i)  %for debuge
    
    if ~isempty(train_data_pos(i).Cart(:))
        img = im2single(imread(train_data_pos(i).imageFilename));
        num_bbox = size(train_data_pos(i).Cart(1:end,:),1);
        
        for j=1:num_bbox
            bbox(j,:) = [train_data_pos(i).Cart(j,1)-2,...
                train_data_pos(i).Cart(j,2)-3,width,height ];
            
            %             img_crop = imcrop(img,bbox(j,:));
            %             img_name = sprintf('Cart_i%d_j%d.jpg', i,j) ;
            %             imwrite(img_crop,img_name,'jpg')
            
            tempImages{i,j} = train_data_pos(i).imageFilename;
            
            tempBoxImages(:,i,j) = [bbox(j,1), bbox(j,2),...
                (bbox(j,1)+width),(bbox(j,2)+height)];
            
            tempBoxLabels{i,j} = 'Cart';
            
            
        end
    end
end

% image path - and labels
UCSD1.Images      = tempImages     (~cellfun('isempty',tempImages));

% bbox [xmin ymin xmax ymax]
UCSD1.BoxImages   = tempBoxImages(:,any(tempBoxImages,1));

% image path
UCSD1.TotalImages2read = unique(UCSD1.Images);


end


