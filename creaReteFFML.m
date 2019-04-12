function reteNeurale = creaReteFFML(numNodiInput,numNodiOutput,funDiAttivazioneOutput,strutturaLivelliHidden,minPeso,maxPeso,numLivHidden)
% creaReteFFML
% Crea una rete neurale feed-forward multi-layer.
% Richiede in input: 
% - numNodiInput: numero dei nodi in input.
% - numNodiOutput: numero dei nodi in output.
% - funDiAttivazioneOutput: puntatore alla funzione di attivazione.
% - strutturaLivelliHidden: array contenente i nodi hidden con relativa funzione di
% attivazione.
% - minPeso: valore minimo di un peso.
% - maxPeso: valore massimo di un peso.
%
% In output restituisce la struttura che descrive la rete neurale.

% Viene uniformata la rappresentazione della matrice
if size(strutturaLivelliHidden,1)>size(strutturaLivelliHidden,2)
    strutturaLivelliHidden=strutturaLivelliHidden';
end

% Inizializzazione struttura output
reteNeurale.numNodiInput=numNodiInput;
reteNeurale.numNodiOutput=numNodiOutput;
reteNeurale.numLivelliHidden=numLivHidden;
reteNeurale.funDiAttivazioneOutput=funDiAttivazioneOutput;
reteNeurale.numLivelli=size(strutturaLivelliHidden,2)+2;
for i=1:numLivHidden
    reteNeurale.m(i)=strutturaLivelliHidden(i).dimLivello;
end

% Generazione casuale dei pesi
% *** PRIMO LIVELLO ***
reteNeurale.b{1}=(maxPeso-minPeso).*rand(1,reteNeurale.m(1))+minPeso;
reteNeurale.W{1}=(maxPeso-minPeso).*rand(reteNeurale.m(1),numNodiInput)+minPeso;
reteNeurale.g{1}=strutturaLivelliHidden(1).funDiAttivazione;
% Generazione casuale dei pesi
% *** LIVELLI HIDDEN ***
if reteNeurale.numLivelliHidden>=2
    for i=2:reteNeurale.numLivelliHidden
        reteNeurale.b{i}=(maxPeso-minPeso).*rand(1,reteNeurale.m(i))+minPeso;
        reteNeurale.W{i}=(maxPeso-minPeso).*rand(reteNeurale.m(i), reteNeurale.m(i-1))+minPeso;
        reteNeurale.g{i}=strutturaLivelliHidden(i).funDiAttivazione;
    end
end
% Generazione casuale dei pesi
% *** ULTIMO LIVELLO ***
reteNeurale.b{reteNeurale.numLivelli-1}=(maxPeso-minPeso).*rand(1,numNodiOutput)+minPeso;
reteNeurale.W{reteNeurale.numLivelli-1}=(maxPeso-minPeso).*rand(numNodiOutput,reteNeurale.m(reteNeurale.numLivelliHidden))+minPeso;
reteNeurale.g{reteNeurale.numLivelli-1}=funDiAttivazioneOutput;
end

