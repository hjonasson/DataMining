
str1 = 'vgpm.m.2013.xyz/vgpm.2013';
str2 = '.all.xyz';
datafiles = ['001', '032', '060', '091', '121', '152', '182', '213', '244', '274', '305', '335'];
slice = 300;
x = -180:180;
y = -90:90;
[X Y] = meshgrid(x,y);
Z = 0;
for k = 1:11
    file = datafiles(k*3-2:k*3)
    mainstr = strcat(str1,file,str2);
    a = importdata(mainstr);
    lon = a.data(:,1);
    lat = a.data(:,2);
    bio = a.data(:,3);
    lon = lon(find(bio ~= -9999));
    lat = lat(find(bio ~= -9999));
    bio = bio(find(bio ~= -9999));
    lon = lon(1:slice:end);
    lat = lat(1:slice:end);
    bio = bio(1:slice:end);
    v = variogram([lon lat],bio,'plotit',true,'nrbins',150);%tweek this paramter a little
    [dum,dum,dum,vstruct] = variogramfit(v.distance,v.val,[],[],[],'model','stable');
    [Zhat,Zvar] = kriging(vstruct,lon,lat,bio,X,Y);
    newfile = strcat(str1,file,'interp',str2);
    fileID = fopen(newfile,'w');
    Z = Z + Zhat;
    for i = 1:361
        for j = 1:181
            zji = Zhat(j,i);
            fprintf(fileID,'%d %d %d\n',x(i),y(j),zji);
        end
    end
    fclose(fileID);
end

Z = Z/12;

fileID = fopen('allBioData.txt','w');
for i = 1:361
    for j = 1:181
        zji = Z(j,i);
        fprintf(fileID,'%d %d %d\n',x(i),y(j),zji);
    end
end
fclose(fileID);
    