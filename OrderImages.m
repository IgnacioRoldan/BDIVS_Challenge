function OrderImages(path)

formatSpec = '%s%s%f%f%f%s%s%f%f%f%f';

T = readtable([path '/Data_Entry_2017.csv'],'Delimiter',',','Format',formatSpec);

for i=1:size(T.Var1,1)
    
    imageName = T.Var1{i};
    labels = strsplit(T.Var2{i}, '|');
    
    for j=1:size(labels,1)
        
        [status, msg, msgID] = mkdir([path '/' labels{j}]);
        
        % If there is only one label move the image. If there is more, the
        % image will bee copied and only moved in the last one.
        if(j==size(labels,1))
            if exist([path '/images/' imageName], 'file') == 2
                movefile([path '/images/' imageName], [path '/' labels{j}])
            end
            
        else
            if exist([path '/images/' imageName], 'file') == 2
                copyfile([path '/images' imageName], [path '/' labels{j}])
            end
        end
    end
    
end
if exist([path '/images'], 'dir') == 7
    rmdir([path '/images']);
end

end
