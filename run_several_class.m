tic;
class_set={'aeroplane_left','aeroplane_right','bicycle_left','bicycle_right',...
    'boat_left','boat_right','bus_left','bus_right','horse_left','horse_right',...
    'motorbike_left','motorbike_right'};
for class_id=1:numel(class_set)
    classname=class_set{class_id};
    class_prepare;
    class_run_evaluate;
    close all;
end
toc;