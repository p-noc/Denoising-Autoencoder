function [reteNeurale,sommatoriaErroriTS,sommatoriaErroriVS] = addestramentoOnline(reteNeurale,trainingSetImg,validationSetImg,trainingSetLabel,validationSetLabel,funErrore,eta,flagSoftmax)
%addestramentoOnline
% Addestra la rete neurale applicando un approccio di tipo online, ovvero
% l'aggiornamento dei pesi avviene durante le iterazioni del ciclo di
% addestramento.
% Ha bisogno in input di:
% - reteNeurale
% - immagini training set
% - etichette training set
% - immagini validation set
% - etichette validation set
% - eta
% - funzione di errore
% - flag per uso softmax
% Restituisce la rete neurale aggiornata e due array contenenti gli errori
% del training set e del validation set

% Inizializzazione variabili per calcolo errore training set e validation
% set
erroreTS=0;
sommatoriaErroriTS=0;
erroreVS=0;
sommatoriaErroriVS=0;

%Implementazione addestramento online
for n=1:size(trainingSetImg,1)
    % Applicazione propagazione in avanti, calcolo errore e sommo errori
    % del validation set
    forwardPropTS=forwardProp(reteNeurale,trainingSetImg(n,:),flagSoftmax);
    erroreTS=funErrore(forwardPropTS.z{forwardPropTS.numLivelliHidden+1},trainingSetLabel(n,:));
    sommatoriaErroriTS=sommatoriaErroriTS+erroreTS;
    
    % Applicazione propagazione in avanti, calcolo errore e sommo errori
    % del validation set
    if (n<=size(validationSetImg,1))
        forwardPropVS=forwardProp(reteNeurale,validationSetImg(n,:),flagSoftmax);
        erroreVS=funErrore(forwardPropVS.z{forwardPropVS.numLivelliHidden+1},validationSetLabel(n,:));
        sommatoriaErroriVS=sommatoriaErroriVS+erroreVS;
    end
    
    % Back propagation su training set
    forwardPropTS=backProp(forwardPropTS,trainingSetLabel(n,:),funErrore);
    % Calcolo derivate parziali bias e pesi
    [derBiasTS,derPesiTS]=derivaPesi(forwardPropTS);
    % Aggiorna pesi rete neurale
    forwardPropTS=aggiornaPesi(forwardPropTS,derBiasTS,derPesiTS,eta);
end
reteNeurale=forwardPropTS;

end

