function [reteNeurale,arrayErroriTS,arrayErroriVS] = addestraRete(reteNeurale,epoche,tipoAddestramento,trainingSetImg,validationSetImg,trainingSetLabel,validationSetLabel,funErrore,eta,flagSoftmax,fissapesi)
%addestraRete
% Addestramento rete neurale con discesa del gradiente

% Inizializzazione strutture di supporto:
    % Array errori training set
    arrayErroriTS=zeros(1,epoche);
    % Array errori validation set
    arrayErroriVS=zeros(1,epoche);

% Condizioni di fermata
    epocheNecessarie=floor(epoche/3);
    reteNeuraleCandidata=reteNeurale;
    erroreVS=realmax;
    
% Implementazione addestramento
for e = 1:epoche
    tempReteNeurale=reteNeurale;
    switch tipoAddestramento
        case 'batch'
            [reteNeurale,arrayErroriTS(e),arrayErroriVS(e)]=addestramentoBatch(reteNeurale,trainingSetImg,validationSetImg,trainingSetLabel,validationSetLabel,funErrore,eta,flagSoftmax,fissapesi);
        case 'online'
            [reteNeurale,arrayErroriTS(e),arrayErroriVS(e)]=addestramentoOnline(reteNeurale,trainingSetImg,validationSetImg,trainingSetLabel,validationSetLabel,funErrore,eta,flagSoftmax);
        otherwise
            error('Tipo di addestramento supportato: [batch][online]');
    end
    contatoreErrore=0;
    % Controllo se l'errore decresce
    if arrayErroriVS(e)<erroreVS
        % Variabile che tiene conto del numero di volte in cui l'errore del
        % validation set cresce.
        contatoreErrore=0;
        erroreVS=arrayErroriVS(e);
        reteNeuraleCandidata=tempReteNeurale;
    else
        % Se l'errore cresce viene aumentato il contatore
        if e>=epocheNecessarie
            contatoreErrore=contatoreErrore+1;
            % Se l'errore cresce troppe volte consecutive superando una
            % soglia arbitraria
            if contatoreErrore>30
                % Viene interrotto il ciclo delle epoche
                break;
            end
        end
    end
    fprintf("epoca corrente :%d\n",e);
end

% Ritorno la rete migliore
reteNeurale=reteNeuraleCandidata;

% All'occorrenza la dimensione degli array di errore viene ridotta
if e<epoche
    arrayErroriTS=arrayErroriTS(1:e);
    arrayErroriVS=arrayErroriVS(1:e);
end
end

