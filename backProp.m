function reteNeurale = backProp(reteNeurale,T,funErrore)
% backProp
% La funzione effettua la propagazione all'indietro della rete, per ogni
% strato vengono calcolati i relativi delta partendo dai nodi di output
% fino a quelli di input.
% Ha bisogno di:
% - reteNeurale
% - T: Target con cui confrontare i valori di output (etichette)
% - funErrore: funzione per il calcolo dell'errore
% Restituisce in output la rete aggiungendo ad essa i delta calcolati
% utilizzando le derivate parziali.
% Per calcolare il delta del livello di output vengono moltiplicate la
% derivata della funzione di attivazione sull'input del nodo di output e la
% derivata parziale della funzione di errore ottenuta confrontando y e t
% (output ultimo nodo e label).

% All'interno di un ciclo verrano calcolati tutti i delta intermedi sui
% livelli hidden, ognuno si otterrà moltiplicando la derivata della
% funzione di attivazione sull'input del nodo ed il prodotto tra la matrice
% dei delta del livello successivo e la matrice dei pesi del livello
% successivo.
% La matrice ottenuta dal prodotto più interno viene poi moltiplicata punto
% per punto con il vettore contente i valori restituiti dalla derivata della 
% funzione di attivazione sull'input di quel livello.

indiceStratoOutput=reteNeurale.numLivelliHidden+1;
reteNeurale.delta{indiceStratoOutput}=reteNeurale.funDiAttivazioneOutput(reteNeurale.a{indiceStratoOutput},true).* funErrore(reteNeurale.z{indiceStratoOutput},T,true);
for i=indiceStratoOutput-1:-1:1
    prodDeltaPesi = reteNeurale.delta{i+1}*reteNeurale.W{i+1}; %calcolo intermedio
    reteNeurale.delta{i} = reteNeurale.g{i}(reteNeurale.a{i},true) .* prodDeltaPesi;
end
end

