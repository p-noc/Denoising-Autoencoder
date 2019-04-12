function [setImmagini,setEtichette,indiceElemUtilizzati] = creaSet(dimensioneSet,indiceElemUtilizzati,matEtichette,matImmagini)
    % creaSet: Partendo da un dataset crea un sotto-insieme, è stata
    % utilizzata per generare training set, validation set e test set.

    % Calcolo dimensione singola partizione
    partizioneCifre = floor(dimensioneSet/10);
    % Inizializzazione strutture di output
    setImmagini=zeros(dimensioneSet,784);
    setEtichette=zeros(dimensioneSet,10);
    % Inizializzazione strutture di supporto
    % Contatore numero di elementi inseriti per ogni cifra
    contCifre=zeros(1,10);
    % Contatore elementi inseriti nell'insieme
    contElementi=0;
    % Finche non viene riempito l'insieme...
    while contElementi < dimensioneSet
        % Prendo una posizione casuale all'interno del dataset
        randPos=floor((60000-1).*rand(1)+1);
        % Controllo che l'elemento non sia già stato inserito
        if indiceElemUtilizzati(randPos)==0
            % Controllo quanti elementi di una cifra ho inserito
            if contCifre(matEtichette(randPos)+1)<partizioneCifre
                % Aggiorni contatori e strutture
                contCifre(matEtichette(randPos)+1)=contCifre(matEtichette(randPos)+1)+1;
                contElementi=contElementi+1;
                indiceElemUtilizzati(randPos)=1;
                % Viene inserito questo elemento nell'insieme
                setEtichette(contElementi,matEtichette(randPos)+1)=1;
                setImmagini(contElementi,:)=matImmagini(randPos,:);
            end
        end
    end
end

