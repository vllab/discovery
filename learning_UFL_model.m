function learning_UFL_model()
% script_siamese_KITTI_ZF()
% Siamese training and testing with part of Zeiler & Fergus model
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%addpath matlab library
addpath(addpath(genpath('../matlablib/edges-master')));
addpath(addpath(genpath('../matlablib/toolbox')));
%% -------------------- CONFIG --------------------
% model
model.cache_name            = 'VGG15W6';
model.mean_image            = fullfile(pwd, 'models','UFL','mean_image.mat');%create by imagemean.m
model.solver_def_file       = fullfile(pwd, 'models','UFL','solver_VGG12.prototxt');%layer we define
data_path='./datasets/VOC2007/train';

% train data
dataset.imdb_train          = imdb_from_voc_hog(data_path,'hog');%load database image path

if(~exist(model.mean_image ,'file'))
    imagemean(dataset.imdb_train,model.mean_image);
end

dataset.pairdb_train        = prepare_hog_similiar(dataset.imdb_train);%prepare proposals pair

% conf
conf                        = siamese_config('image_means', model.mean_image);%network config
%% -------------------- TRAIN --------------------
fprintf('\n***************\ntraining stage \n***************\n');    
% try to find trained model
imdbs_name = dataset.pairdb_train.name;
cache_dir = fullfile(pwd, 'output', 'hyper_hog', model.cache_name, imdbs_name);%the path to save model
save_model_path = fullfile(cache_dir, 'final');
if exist(save_model_path, 'file')%do not train model again if exist
    return;
end

% init caffe solver
mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');%log file path
caffe.init_log(caffe_log_file_base);
caffe_solver = caffe.Solver(model.solver_def_file);
% caffe_solver.net.copy_from(model.init_net_file);
% we do not use other model to pretrain

% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'log'));
log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
diary(log_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

disp('conf:');
disp(conf);

% training
shuffled_inds = [];
train_results = [];
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();

conf_metric=conf;
save(fullfile(cache_dir,'conf.mat'),'conf_metric','-v7.3');

warning off;
figure;colormap('jet');
while (iter_ < max_iter)
    caffe_solver.net.set_phase('train');
    
    % generate minibatch training data
    [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, dataset.pairdb_train, conf.ims_per_batch);
    [net_inputs,im] = pair_generate_minibatch(conf, dataset.pairdb_train.rois, sub_db_inds);
    net_input{1} = net_inputs{1};
    idx=randperm(4);
    for i=1:4
        pair=idx(i);
        if(pair<3)
            net_input{2}=net_inputs{2};
        else
            net_input{2}=net_inputs{3};
        end
        %label
        tmp=net_inputs{4};
        tmp=tmp(:,:,:,pair);
        net_input{3}=tmp;
        %roi
        tmp=net_inputs{5};
        tmp=tmp(:,:,:,pair);
        net_input{4}=tmp;
        %roip
        tmp=net_inputs{6};
        tmp=tmp(:,:,:,pair);
        net_input{5}=tmp;

        caffe_solver.net.reshape_as_input(net_input);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_input);
        caffe_solver.step(1);
        rst = caffe_solver.net.get_output();
        rst=[rst struct('blob_name', 'label', 'data', pair)];
        train_results = parse_rst(train_results, rst);
        if(i==1)
            %show the training view
            subplot(1,3,1);
            imshow(im);
            axis image;
            map=caffe_solver.net.blobs('conv3_2').get_data();%load conv3_2 layer
            subplot(1,3,2);
            map1=sum(map,3)';
            imagesc(map1);axis image;
            m=reshape(map,[],64);%every 4 channel show one response
            m=mean(m);
            subplot(2,3,3);
            imagesc(m(1:32));
            colorbar;
            subplot(2,3,6);
            imagesc(m(33:end));
            colorbar;
            drawnow;
        end
            % do valdiation per val_interval iterations
        if ~mod(iter_, conf.val_interval)        
            show_state(iter_, train_results);
            train_results = [];
            diary; diary; % flush diary
        end

        % snapshot
        if ~mod(iter_, conf.snapshot_interval)
            snapshot(caffe_solver, cache_dir, sprintf('iter_%d', iter_));

        end

        iter_ = caffe_solver.iter();
    end
end

    % final snapshot
    snapshot(caffe_solver, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(caffe_solver, cache_dir, 'final');

diary off;
caffe.reset_all();
rng(prev_rng);
end

function conf = siamese_config(varargin)

    ip = inputParser; 
    
    % whether use gpu
    ip.addParamValue('use_gpu',           gpuDeviceCount > 0,   @islogical);
    % random seed
    ip.addParamValue('rng_seed',          5,                    @isscalar);
    % Images per batch
    ip.addParamValue('ims_per_batch',     1,                    @isscalar);% take care      
    % Image scales -- the short edge of input image                                                
    ip.addParamValue('scales',            600,                  @ismatrix);    
    % Max pixel size of a scaled input image
    ip.addParamValue('max_size',          1000,                 @isscalar);
    % Iteration times of validation
    ip.addParamValue('val_iters',         500,                  @isscalar);
    % Interval of validation
    ip.addParamValue('val_interval',      100,                  @isscalar);
    % Interval of snapshot
    ip.addParamValue('snapshot_interval', 10000,                @isscalar);
    % mean image, in RGB order
    ip.addParamValue('image_means',       128,                  @ismatrix);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, pairdb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % random perm
        lim = floor(numel(pairdb_train.rois) / ims_per_batch) * ims_per_batch;
        pair_inds = randperm(numel(pairdb_train.rois), lim);
        pair_inds = reshape(pair_inds,ims_per_batch,[]);
        shuffled_inds = num2cell(pair_inds, 1);
    end

    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function [input_blobs,im] = pair_generate_minibatch(conf, pairdb, inds)
    % build the region of interest and label blobs
    roi_blob = zeros(0, 5, 'single');
    roip_blob = zeros(0, 5, 'single');
    label_blob = zeros(0, 1, 'single');
    % Get the input image blob
    ind=inds;
    box_id=[];
    box_id_p=[];
    box_id_dis=[];
    %load image bbox      
    idx=randi(size(pairdb(inds).pair,1));%random sample idx
    label_blob=[label_blob;1;0;0;0];%one positive and three negative
    ind_p=pairdb(inds).idx(idx);
    %neighboring image data
    images={pairdb(ind).image};
    box_id=[box_id;repmat(pairdb(ind).target(idx),1,2)];

    images_p={pairdb(ind_p).image};
    ii=randi(3)+1;
    box_id_p = [box_id_p;pairdb(ind).pair(idx,[1 ii])];%pair box id
    %random image data
    ind_dis=randi(numel(pairdb));
    images_dis={pairdb(ind_dis).image};
    desc1=pairdb(ind).desc;
    desc2=pairdb(ind_dis).desc;
    D=desc1'*desc2;
    [value,list]=sort(D,2);
    [~,idx]=sort(value(:,1));
    
    idx1=idx(1);%first dislike
    idx2=list(idx1,1);

    idx_1=idx(2);%secend dislike
    idx_2=list(idx_1,1);
    
    box_id=[box_id idx1 idx_1];
    box_id_dis=[idx2 idx_2];
    
    %prepare caffe type
        
    [im_blob, im_scales] = get_image_blob(conf,images);
    [im_blob_p, im_scales_p] = get_image_blob(conf,images_p);
    [im_blob_dis, im_scales_dis] = get_image_blob(conf,images_dis);
    
    batch_ind = ones(4,1);
    %union1
    roi   = round((pairdb(ind).boxes(box_id, :)-1) * im_scales) + 1; % in matlab's index (start from 1)
    rois_blob_this_image = [batch_ind, roi];
    roi_blob = [roi_blob; rois_blob_this_image];
    %union2
    roi   = round((pairdb(ind_p).boxes(box_id_p, :)-1) * im_scales_p) + 1; % in matlab's index (start from 1)
    %union3
    roi2   = round((pairdb(ind_dis).boxes(box_id_dis, :)-1) * im_scales_dis) + 1; % in matlab's index (start from 1)

    rois_blob_this_image = [batch_ind, [roi;roi2]];
    roip_blob = [roip_blob; rois_blob_this_image];

   
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    im_blob_p = im_blob_p(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob_p = single(permute(im_blob_p, [2, 1, 3, 4]));
    im_blob_dis = im_blob_dis(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob_dis = single(permute(im_blob_dis, [2, 1, 3, 4]));
    label_blob = single(permute(label_blob, [3, 4, 2, 1]));
    roi_blob = roi_blob - 1; % for c's index start from 0
    roi_blob = single(permute(roi_blob, [3, 4, 2, 1]));
    roip_blob = roip_blob - 1; % for c's index start from 0
    roip_blob = single(permute(roip_blob, [3, 4, 2, 1]));
    
    assert(~isempty(im_blob));
    assert(~isempty(im_blob_p));
    assert(~isempty(label_blob));
    assert(~isempty(roi_blob));
    assert(~isempty(roip_blob));
    
    im=imread(pairdb(ind).image);
    input_blobs = {im_blob ,im_blob_p,im_blob_dis, label_blob, roi_blob ,roip_blob};
end

function [im_blob, im_scales] = get_image_blob(conf,images)

%     im = imread(pairdb.image{inds});
%     target_size = conf.scales;
%     
%     [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
%     
%     im_scales = im_scale;
%     processed_im = {im};
%     
%     im_blob = im_list_to_blob(processed_im);
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    target_size = conf.scales;
    for i = 1:num_images
        im = imread(images{i});
        
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

function show_state(iter, train_results)

    mergin=1.7;
    label_true=train_results.label.data==1;
    label_false=train_results.label.data==2;
    label_neg1=train_results.label.data==3;
    label_neg2=train_results.label.data==4;
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training: loss : (%.3g)\n',mean(train_results.contrastiveloss.data));
    a6=mean(train_results.contrastiveloss.data(label_true));
    b6=mean(train_results.contrastiveloss.data(~label_true));
     fprintf('label true : (%.3g) false : (%.3g)\n',a6,b6);
    dist=train_results.contrastiveloss.data(label_true);
    dist=sqrt(dist)/2;
    sim_angle=180*2*asin(dist)/pi;
    dist=train_results.contrastiveloss.data(label_false);
    dist=(mergin-sqrt(dist))/2;
    dis_angle=180*2*asin(dist)/pi;
    dist=train_results.contrastiveloss.data(label_neg1);
    dist=(mergin-sqrt(dist))/2;
    neg1_angle=180*2*asin(dist)/pi;
    dist=train_results.contrastiveloss.data(label_neg2);
    dist=(mergin-sqrt(dist))/2;
    neg2_angle=180*2*asin(dist)/pi;
    fprintf('angle similiar : (%.3g) dissimiliar : (%.3g) neg1 : (%.3g) neg2 : (%.3g)\n', ...
        mean(sim_angle),mean(dis_angle),mean(neg1_angle),mean(neg2_angle));
end

function model_path = snapshot(caffe_solver, cache_dir, file_name)

    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);
end