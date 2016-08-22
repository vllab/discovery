function [Smap,bdIds,geoDist]=Do_Saliency(adjcMatrix, pixelList,posDistM,colDistM,bg_map,bdIds,clipVal)
%% prepare our bdids
    fg_map=~bg_map;
    for i=1:numel(pixelList)
        id=pixelList{i};
        N=sum(fg_map(id))/(numel(id));
        if(N<0.75)
            bdIds=[bdIds;i];
        end
    end
    bdIds=unique(bdIds);

%% Geodesic Saliency
    geoDist = GeodesicSaliency(adjcMatrix, bdIds, colDistM, posDistM, clipVal);
    Smap=zeros(size(bg_map));
    for i=1:numel(pixelList)
        id=pixelList{i};
        Smap(id)=geoDist(i);
    end
end
