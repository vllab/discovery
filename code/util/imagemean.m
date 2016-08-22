function imagemean(imdb,save_path)
%% Compute mean
N=0;
img_sum=zeros(1,1,3);
for i = 1:numel(imdb.image_ids)
    tic_toc_print('read : %d/%d\n', i, numel(imdb.image_ids));
    I = imread(imdb.image_at(i));
    [w,h,~]=size(I);
    NN=w*h;
    img_sum = img_sum + double(sum(reshape(I,[],1,3)));
    N=N+NN;
end

img_mean = single(img_sum/N);
save(save_path, 'img_mean');