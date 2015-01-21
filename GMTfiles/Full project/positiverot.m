%
% Calculates the extra rotation needed to bring composition % scores into a positive space from qmode2mainm.
%
function[ERCS]=positiverot(X,CFS)
% CFS:Composition factor scores resulting after % "composition scores function"
% X:matrix of original data
% ERCS:Extra rotated composition scores
meanx=mean(X); [nv,nf]=size(CFS); MaxRT=ones(1,nf); ERCS=ones(nv,nf);
for j=1:nf MaxR=0;
for i=1:nv
if CFS(i,j)<0 Rot=CFS(i,j)/(CFS(i,j)- meanx(i)); if Rot>MaxR
MaxR=Rot;
end
end
end MaxRT(j)=MaxR; if MaxRT(j)>0
for i=1:nv ERCS(i,j)=((1-MaxRT(j))*CFS(i,j))+((MaxRT(j)*meanx(i))); if ERCS(i,j)<0
ERCS(i,j)=0; end
end
elseif MaxRT(j)==0
for i=1:nv ERCS(i,j)=CFS(i,j);
end end
end