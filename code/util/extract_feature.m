function [feat,data]= extract_feature(conf, caffe_net, im, boxes,varargin)
%varargin : other layer data output
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, im_scale] = get_image_blob(conf, im);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    
    rois_blob =bbox_to_blobs(conf,boxes,im_scale);
    
    net_inputs = {im_blob rois_blob};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    feat = caffe_net.forward(net_inputs);
    N=length(varargin);
    if(N>0)
        data=cell(1,N);
        for i=1:N
            data{i}=caffe_net.blobs(varargin{i}).get_data();
        end
    end
end
function rois_blob =bbox_to_blobs(conf,boxes,im_scale)
    [rois_blob, ~] = get_blobs(conf,boxes,im_scale);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
end
function [rois_blob,im_scale] = get_blobs(conf, rois,im_scale)
    rois_blob = get_rois_blob(conf, rois,im_scale);
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(~, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end

function [im_blob, im_scale] = get_image_blob(conf,image)
    [im_blob, im_scale] = prep_im_for_blob(image, conf.image_means, conf.test_scales, conf.max_size); 
end
