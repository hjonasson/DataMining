%
% Formats and outputs the factor loadings from qmode2mainm. %
function x = bmatrixprint(lpfid,title,inmatrix) [n m] = size(inmatrix);
rownum = [1: 1:n]';
label = '';
for i = 1: m-2
label = [label '%8i ']; end
label = [label '\n']; format = '%8i %8i '; for j = 1: m-1
format = [format '%8.4f ']; end
format = [format,'\n'];
fprintf(lpfid,[title ' \n']);
fprintf(lpfid,[' No. Sample ID Comm. ', label], [1: 1:m-2]); fprintf(lpfid,format,[rownum inmatrix]');
fprintf(lpfid,'\n');