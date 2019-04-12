function out = ReLU(x,daDerivare)
%Funzione di attivazione rectified linear unit.
if exist('daDerivare','var')
    if x < 0
        out=0;
    else
        out=1;
    end
else
    if x < 0
        out=0;
    else
        out=x;
    end
end
end