function [derBias,derPesi] = derivaPesi(reteNeuraleBP)
% derivaPesi
% La funzione calcola, per ogni livello, le derivate parziali della
% funzione di errore rispetto ai pesi e bias del livello.
% Richiede in input:
% - reteNeuraleBP: rete neurale sulla quale è stata effettuato il calcolo 
% della back propagation.
% In output restituisce i vettori contententi le derivate dei pesi e dei bias.

% Inizialmente vengono calcolate le derivate parziali del primo livello
% hidden. Per i bias le derivate vengono sommate tra loro, mentre per i
% pesi è necessario effettuare un prodotto tra la matrice dei delta e X
% (input). Per effettuare il prodotto la matrice dei delta viene trasposta.

% Per i restanti livelli (hidden+output) il calcolo verrà effettuato allo
% stesso modo (tra il livello i e i-1)

    derBias{1}=sum(reteNeuraleBP.delta{1},1);
    derPesi{1}=reteNeuraleBP.delta{1}' * reteNeuraleBP.X;

    for i=2 : reteNeuraleBP.numLivelliHidden+1
        derBias{i}=sum(reteNeuraleBP.delta{i},1);
        derPesi{i}=reteNeuraleBP.delta{i}' * reteNeuraleBP.z{i-1};
    end
end

