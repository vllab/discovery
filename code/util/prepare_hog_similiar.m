function roidb=prepare_hog_similiar(imdb)
    roidb.name = imdb.name;%database name
    %prepare data information
    cache_file = fullfile(['./imdb/cache/roidb_' imdb.name]);
    cache_file = [cache_file '.mat'];%%the path to save information
    if exist(cache_file, 'file')%if exist information , do loading
        load(cache_file);
    else
        %prepare information
        roidb.name = imdb.name;
        roidb.rois = get_similar(imdb);%get similar data
        fprintf('Saving roidb to cache...');
        save(cache_file, 'roidb', '-v7.3');
        fprintf('done\n');
    end
