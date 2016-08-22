function database=dirimg(rt_img_dir,pic_t)
%==========================================================================
% usage: calculate the sift descriptors given the image directory
%
% inputs
% rt_img_dir    -image database root path
% rt_data_dir   -feature database root path
% gridSpacing   -spacing for sampling dense descriptors
% patchSize     -patch size for extracting sift feature
% maxImSize     -maximum size of the input image
% nrml_threshold    -low contrast normalization threshold
%
% outputs
% database      -directory for the calculated sift features
%
% Lazebnik's SIFT code is used.
%
% written by Jianchao Yang
% Mar. 2009, IFP, UIUC
%==========================================================================

subfolders = dir(rt_img_dir);

database = [];

database.imnum = 0; % total image number of the database
database.img = {};
if( nargin<2)
    pic_t='*.jpg';
end
frames = dir(fullfile(rt_img_dir,pic_t));
c_num = length(frames);  
database.imnum =+ c_num;

for jj = 1:c_num,
    imgpath = fullfile(rt_img_dir,frames(jj).name);
    database.img = [database.img, imgpath];
end