function [rispostaClasse] = adattaRisposta(Y)
%adattaRisposta: 
%Converte nel formato scelto la classificazione effettuata dalla
%rete.
%Richiede in input:
%- la matrice di output della rete
%Restituisce in output:
%- una matrice contenente l'output della rete adattato, l'elemento (i,j)
%contiene l'i-esimo dato in input per la classe j.

rispostaClasse = zeros(size(Y,1), size(Y,2));
for i = 1 : size(Y,1)
    %viene individuata la classe per la quale la rete ha fornito la
    %risposta maggiore.
    [~, argmax] = max(Y(i,:));
    %adatto al formato della risposta
    rispostaClasse(i,argmax) = 1;
end
end

