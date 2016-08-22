close all;
clc;
%add matlab library
addpath(addpath(genpath('../matlablib/edges-master')));
addpath(addpath(genpath('../matlablib/toolbox')));

%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id);

opts.use_gpu                = true;

%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(pwd, 'models','UFL');%model path
model    = load(fullfile(model_dir, 'conf'));% loading other config

model.metric_net_def = fullfile(model_dir,'test.prototxt');%model prototxt
model.metric_net = fullfile(model_dir,'iter_150000');%model parameter

model.conf_metric.test_scales = model.conf_metric.scales;
if opts.use_gpu
    model.conf_metric.image_means = gpuArray(model.conf_metric.image_means);
end

% proposal net
metric_net = caffe.Net(model.metric_net_def, 'test');
metric_net.copy_from(model.metric_net);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end    