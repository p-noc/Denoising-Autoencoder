function [reteNeurale,erroreTS,erroreVS] = addestramentoBatch(reteNeurale,trainingSetImg,validationSetImg,trainingSetLabel,validationSetLabel,funErrore,eta,flagSoftmax,fissapesi)
% addestramentoBatch
% Addestra la rete neurale applicando un approccio di tipo batch, ovvero
% l'aggiornamento dei pesi avviene alla fine, dopo aver calcolato l'errore.
% Ha bisogno in input di:
% - reteNeurale
% - trainingSetImg
% - validationSetImg
% - trainingSetLabel
% - validationSetLabel
% - funErrore, funzione di errore utilizzata per l'addestramento
% - eta, costante moltiplicativa utilizzata nella discesa del gradiente
% - flagSoftMax, se flag vera viene utilizzato softmax
% - fissapesi, se flag utilizzata per bloccare i pesi del primo livello
% dell'autoencoder a due livelli

% Inizializzazione variabili per calcolo errore training set e validation set
erroreTS=0;
erroreVS=0;

% Applicazione propagazione in avanti per training set e validation set
forwardPropTS=forwardProp(reteNeurale,trainingSetImg,flagSoftmax);
forwardPropVS=forwardProp(reteNeurale,validationSetImg,flagSoftmax);

% Calcolo errore training set e validation set, vengono divisi per la 
erroreTS=sum(funErrore(forwardPropTS.z{forwardPropTS.numLivelliHidden+1},trainingSetLabel))/size(trainingSetImg,1);
erroreVS=sum(funErrore(forwardPropVS.z{forwardPropVS.numLivelliHidden+1},validationSetLabel))/size(validationSetImg,1);
% Back propagation su training set
forwardPropTS=backProp(forwardPropTS,trainingSetLabel,funErrore);
% Calcolo derivate parziali bias e pesi
[derBiasTS,derPesiTS]=derivaPesi(forwardPropTS);
% Aggiorna pesi rete neurale
forwardPropTS=aggiornaPesi(forwardPropTS,derBiasTS,derPesiTS,eta,fissapesi);

reteNeurale=forwardPropTS;
end

