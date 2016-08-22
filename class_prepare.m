%data info
imdir=['./datasets/voc07_6X2/imgs/' classname];%image path
gt=get_gt_from_gtbbs(['./datasets/voc07_6X2/gt/' classname]);
%loadground truth ,one image ground truthis one cell
%one cell have several box [xmin ymin xmax ymax]

%edge box setting
database= dirimg(imdir);
edge_model=load('edge/modelBsds');
edge_model=edge_model.model;
edge_model.opts.multiscale=0; edge_model.opts.sharpen=2; edge_model.opts.nThreads=4;
edge_opts = edgeBoxes;
edge_opts.alpha = .65;     % step size of sliding window search
edge_opts.beta  = .75;     % nms threshold for object proposals
edge_opts.minScore = .01;  % min score of boxes to detect
edge_opts.maxBoxes = 1e4;  % max number of boxes to detect

selectrate=1000;
data=struct('desc',[],'feat',[],'box',[],'relation',[],'IOU',[],'H',1,'W',1,'img','N');
D=data;

for ind=1:database.imnum
        %read image
        img=imread(database.img{ind});
        [H,W,~]=size(img);
        im=img;
        if opts.use_gpu
            im = gpuArray(im);
        end
        %edge-boxes
        bbs=double(edgeBoxes(img,edge_model,edge_opts));
        if(size(bbs,1)>selectrate)
            bbs(selectrate+1:end,:)=[];
        end
        move=bbs(:,5);
        bbs(:,5)=[];
        box=bbs;
        box(:,3:4)=bbs(:,3:4)+bbs(:,1:2);
        box(end+1,:)=[1 1 W H];
        %     feature get
        feat= extract_feature(model.conf_metric, metric_net, im, box);%[xmin, ymin, xmax, ymax]
        D.desc=feat{1}(:,1:end-1);%bounding box feature
        D.feat=feat{1}(:,end);%entire image feature
        
        D.box=bbs;
        %relationship
        l_center=bbs(:,1:2)+bbs(:,3:4)/2;
        d_l_move=[];
        d_l_move(:,:,1)=bsxfun(@minus,l_center(:,1),l_center(:,1)');
        d_l_move(:,:,2)=bsxfun(@minus,l_center(:,2),l_center(:,2)');
        dsx=bsxfun(@plus,bbs(:,3),bbs(:,3)')./2;
        dsy=bsxfun(@plus,bbs(:,4),bbs(:,4)')./2;
        d_l_move(:,:,1)=bsxfun(@rdivide,d_l_move(:,:,1),dsx);%(cx-cx')/(w+w')
        d_l_move(:,:,2)=bsxfun(@rdivide,d_l_move(:,:,2),dsy);%(cy-cy')/(h+h')
        d_l_move(:,:,end+1)=log(bsxfun(@rdivide,bbs(:,3),bbs(:,3)'))./log(2);%log(w/w')
        d_l_move(:,:,end+1)=log(bsxfun(@rdivide,bbs(:,4),bbs(:,4)'))./log(2);%log(h/h')
        D.relation=d_l_move;
        D.IOU=cmp_IOU(bbs);
        D.H=H;
        D.W=W;
        D.img=database.img{ind};
        data(ind)=D;
        tic_toc_print('feat : %d/%d\n', ind,database.imnum);
end
clearvars -except model metric_net edge_model edge_opts data opts classname gt class_set class_id;