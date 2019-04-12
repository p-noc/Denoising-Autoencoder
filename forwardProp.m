function reteNeurale = forwardProp(reteNeurale,X,flagSoftmax)
% forwardProp
% Propagazione in avanti della rete neurale
% Prende come input:
% - reteNeurale
% - X: matrice contenente le immagini di input
% - flagSoftmax: flag per stabilire se applicare softmax 
% Restituisce:
% - a: valori di input ricevuti dell'intero layer i dato il vettore di input j
% - z: output dei nodi
% - X: come sopra

% Inizializzo strato di input
reteNeurale.X=X;
% Array dei valori di output del livello precedente 
zPrec=X; % Al primo livello coincide con l'input

% Per ogni livello (escluso input) viene calcolato l'input e l'output del
% livello
for i=1 : reteNeurale.numLivelliHidden+1
    % Per calcolare la a (input) associata al livello corrente avremo da 
    % considerare un numero di pesi pari al prodotto tra i nodi di questo
    % livello e i nodi del livello precedente. 
    % Facendo il prodotto tra la matrice dei valori correnti trasposta e 
    % gli output del livello precedente si ottiene una matrice che avrà
    % come righe il numero di nodi di input sul layer attuale e come colonne 
    % il numero di vettori per cui è stata moltiplicata.
    % Per uniformarsi agli standard il prodotto finale viene trasposto.
    % Per calcolare invece la z (output) associata al livello corrente 
    % abbiamo bisogno di sommare il bias del layer corrente ad ogni riga
    % della matrice di input.
    
    reteNeurale.a{i}=(zPrec*reteNeurale.W{i}');
    reteNeurale.a{i}=reteNeurale.a{i}+reteNeurale.b{i};
    reteNeurale.z{i}=reteNeurale.g{i}(reteNeurale.a{i});
    zPrec=reteNeurale.z{i};
end

% Controllo flag softmax
if flagSoftmax
    % Il calcolo del softmax è stato implementato seguendo le indicazioni
    % date durante il corso. 
    softmax=exp(reteNeurale.z{reteNeurale.numLivelliHidden+1})./ sum(exp(reteNeurale.z{reteNeurale.numLivelliHidden+1}),2);
    reteNeurale.z{reteNeurale.numLivelliHidden+1}=softmax;
end

