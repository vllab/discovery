init_neighbor_num=10;
img_num=numel(data);
updata_iter=3;%update time
Ntop=60;%believe how many bounding box

figure;color_L=colormap('jet');
class_S=zeros(img_num,1);%result

path_result = fullfile('result', classname);
if isempty(dir(path_result))
    mkdir(path_result);
end
%find neighbor
feat=cell2mat({data.feat})';
D=tool_dist(feat,feat);
D=-D;
D=D-diag(ones(img_num,1)*inf);% a image can not be itself neighboring image
[NB_score,NB_ind]=sort(D,2,'descend');
NB_idx=num2cell(NB_ind(:,1:init_neighbor_num),2);
%find best subgraph
for i=1:img_num
    NB_index=NB_idx{i};
    target=data(i);
    % Saliency
    noFrameImg = imread(target.img);
    [h, w, chn] = size(noFrameImg);
    frameRecord = [h, w, 1, h, 1, w];
    % Segment input rgb image into patches (SP/Grid)
    pixNumInSP = 600;                           %pixels in each superpixel
    spnumber = round( h * w / pixNumInSP );     %super-pixel number for
    [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);%calculate superpixels
    meanRgbCol = GetMeanColor(noFrameImg, pixelList);%superpixels RGB 
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);%superpixels LAB
    meanPos = GetNormedMeanPos(pixelList, h, w);%superpixels position 
    colDistM = GetDistanceMatrix(meanLabCol);%RGB distance
    posDistM = GetDistanceMatrix(meanPos);%position distance
    bdIds = GetBndPatchIds(idxImg);% periphery superpixels of image
    [clipVal, ~, ~] = EstimateDynamicParas(adjcMatrix, colDistM);
    %estimate relation between any two bounding box
    fg_idx=[];%neighboring image select region
    for j=1:numel(NB_index)
        idx=NB_index(j);
        source=data(idx);
        [score,T_i]=best_pair(source,target,Ntop,3);
        
        X=target.IOU*score;%bounding score
        [~,ind]=max(X);%select the highest score bounding box
        fg_idx=[fg_idx ind];
    end
    m=get_fg(target.H,target.W,target.box,fg_idx);% get foreground value
    R=sum(m(:))/max(m(:));
    R=sqrt(R)/5;%smooth
    
    filter=fspecial('disk',R);
    m=conv2(m,filter,'same');
    
    bg_map=m<=(max(m(:))/4); %background is on value area
    [map,bdIds_c]=Do_Saliency(adjcMatrix, pixelList,posDistM,colDistM,bg_map,bdIds,clipVal);
    id_map=zeros(size(bg_map));%background map
    for k=1:numel(bdIds_c)
        ii=bdIds_c(k);
        id=pixelList{ii};
        id_map(id)=1;
    end
    
    Tmap=map>(max(map(:))/10)&id_map==0;%cut Saliency
    %choose big one
    box= regionprops(Tmap,'Boundingbox');
    B=ceil(cell2mat({box.BoundingBox}'));
    BB=B(:,3).*B(:,4);
    [~,ii]=max(BB);
    box=B(ii,:);
    
    tic_toc_print('compute : %d/%d\n', i,img_num);
    %see heat map
    img=imread(target.img);
    subplot(2,3,1)
    imshow(img);axis image;

    subplot(2,3,2);
    imagesc(m);axis image;
    xlabel('area map');

    subplot(2,3,3);
    imagesc(id_map);axis image;
    xlabel('background map');

    subplot(2,3,4);
    imagesc(map);axis image;
    xlabel('Saliency');

    subplot(2,3,5);
    imagesc(Tmap);axis image;
    xlabel('foreground label');

    subplot(2,3,6);
    imshow(img);axis image;
    gt_box=gt{1,i};
    gt_box(:,3:4)=gt_box(:,3:4)-gt_box(:,1:2);
    box1=[box;gt_box];
    IOU=cmp_IOU(box1);
    [SS,idx]=max(IOU(1,2:end));
    rectangle('Position',box,'linestyle','--','EdgeColor','blue','LineWidth',2);
    rectangle('Position',gt_box(idx,:),'EdgeColor','red','LineWidth',2);
    xlabel(['find best bounding box ' num2str(SS)]);
    class_S(i)=SS;
    saveas(gcf,fullfile(path_result, sprintf('sai%03d.jpg', i)));
end
figure;plot(sort(class_S,1,'descend'));
N=sum(class_S>0.5)/numel(class_S);
xlabel([classname ' ' num2str(N)]);
saveas(gcf,fullfile(path_result,'result.jpg'));
res=sort(class_S,1,'descend');
save(fullfile(path_result,'res.mat'),'res');