function reteNeurale = aggiornaPesi(reteNeurale,derBias,derPesi,eta,fissaPesiPrimoLivello)
% AggiornaPesi
% La funzione aggiorna pesi e bias della rete utilizzando i valori
% calcolati precedentemente tramite le derivate. 
% L'aggiornamento sarà modulato dal parametro eta che regolerà l'intervallo
% di scarto da considerare tra un valore candidato e il successivo
% (spostamento sul grafico della curva).
% Prende in input:
% - reteNeurale
% - derivate bias
% - derivate pesi
% - eta
% - flag fissaPesiPrimoLivello
% Restituisce in output la rete neurale con pesi e bias aggiornati.

for i=1:reteNeurale.numLivelliHidden+1
    % se flag fissapesi attiva, non vengono aggiornati i pesi del primo
    % livello della rete
    if fissaPesiPrimoLivello 
        if (i ~= 1)
            reteNeurale.b{i}=reteNeurale.b{i}-(eta*derBias{i});
            reteNeurale.W{i}=reteNeurale.W{i}-(eta*derPesi{i});
        end
    else
        reteNeurale.b{i}=reteNeurale.b{i}-(eta*derBias{i});
        reteNeurale.W{i}=reteNeurale.W{i}-(eta*derPesi{i});
    end
end
end

