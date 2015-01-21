%
% Formats and outputs data matrices %
function x = matrixprint(lpfid,title,inmatrix) [n m] = size(inmatrix);
rownum = [1: 1:n]';
label = '';
for i = 1: m
label = [label '%8i ']; end
label = [label '\n']; format = '%8i %8.4f '; for j = 1: m-1
format = [format '%8.4f ']; end
format = [format,'\n'];
fprintf(lpfid,[title ' \n']); fprintf(lpfid,['Row No.', label], [1: 1:m]); fprintf(lpfid,format,[rownum inmatrix]'); fprintf(lpfid,'\n');