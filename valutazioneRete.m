function [accuratezza] = valutazioneRete(Y,T)
%valutazioneRete: funzione che valuta l'accuratezza complessiva della rete
%nel classificare le cifre.
%richiede in input:
%- la matrice di output Y, contenente il risultato restituito dalla rete
%- T, matrice contenente le label associate al set%
%restituisce in output: 
% -L'accuratezza complessiva della rete nella classificazione.

%Check degli input
if(size(Y,1) ~= size(T,1)) || (size(Y,2) ~= size(T,2))
    error("Le dimensioni dei parametri non coincidono");
end

%ricavo la risposta della rete
rispostaClasse = adattaRisposta(Y);

%viene effettuato il calcolo del numero ri risposte corrette
corrette = nnz(rispostaClasse .* T);

%e viene calcolata l'accuratezza 
accuratezza = corrette/size(Y,1);

end