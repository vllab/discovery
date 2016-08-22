function imdb = imdb_from_voc_hog(root_dir, image_set)

cache_file = ['./imdb/cache/imdb_' image_set '.mat'];
if exist(cache_file, 'file')%if exist information , do loading
	load(cache_file);
else
  imdb.name = image_set;
  imdb.image_dir = root_dir;
  imdb.image_ids = textread(fullfile(root_dir,'train.txt'), '%s');%rewrite
  imdb.extension = 'jpg';
  imdb.roidb_func = @roidb_from_voc07;
  imdb.image_at = @(i) ...
      sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);

  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    info = imfinfo(imdb.image_at(i));
    imdb.sizes(i, :) = [info.Height info.Width];
  end
  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end
