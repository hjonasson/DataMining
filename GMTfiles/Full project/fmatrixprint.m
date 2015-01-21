%
% Formats and outputs the factor scores from qmode2mainm. %
function x = fmatrixprint(lpfid,title,inmatrix,variable) [n m] = size(inmatrix);
rownum = [1: 1:n]';
label = '';
for i = 1: m
label = [label '%12i ']; end
label = [label '\n']; format = '%12.4f '; for j = 1: m-1
format = [format '%12.4f ']; end
format = [format,'\n'];
fprintf(lpfid,[title ' \n']);
fprintf(lpfid,[' Row No. Variable', label], [1: 1:m]); for i = 1: n
lineout = sprintf('%12i ',rownum(i));
lineout = [lineout, sprintf(' %s ',variable(i).names)]; lineout = [lineout, sprintf(format,inmatrix(i,:))]; %fprintf(lpfid,format,[rownum inmatrix]'); fprintf(lpfid,'%s',lineout);
end
fprintf(lpfid,'\n');