% mainScript:
% Questo script viene utilizzato per addestrare una rete utilizzando il 
% criterio di minimizzazione dell'errore della discesa del gradiente.
% Successivamente viene addestrato un autoencoder e viene testata la rete
% con input uguale all'uscita dell'autoencoder applicato ad un dataset
% con rumore.

% Vengono cancellate le variabili d'ambiente di eventuali esecuzioni precedenti
clearvars;
tic
% Aggiunge al workspace il percorso alla cartella resources. 
% Al suo interno:
% - le funzioni di attivazione 
% - il dataset MINST (immagini e label)
addpath('./risorse/');

% Funzione di attivazione dei nodi di output
Class_funNodiOutput = @funIdentita;
% Funzione di attivazione dei nodi interni 
Class_funNodiHidden = @funSigmoide;
% Funzione d'errore
Class_funErrore = @funCrossEntropy;
% Numero di nodi interni
Class_nodiHidden = 180;
% Numero di epoche
Class_epoche = 300;
% Tipo di addestramento: "online" o "batch", altre stringhe sono causa
% d'errore
tipoAddestramento = "batch";
% Numero di elementi del training set
dimensioneTrainingSet = 1000;
% Numero di elementi del test set
dimensioneTestSet = 250;
% Numero di elementi del validation set
dimensioneValidationSet = 250;
% Valore casuale minimo del peso
ClassminPeso = -0.09;
% Valore casuale massimo del peso
ClassmaxPeso = 0.09;
%numero di livelli hidden della rete
ClasslvHidden = 1;
% Valore eta, utilizzato per la discesa del gradiente
%migliore era 0.0004
Class_eta = 0.0004;
% Softmax, funzione esponenziale normalizzata, utilizzabile per rimappare
% l'intervallo di classificazione in modo da utilizzare la sigmoide per
% problemi con più di due classi
flagSoftmax = true;

% Tipo rumore utilizzato, 
% sono supportati  -> Standard rumore utilizzato nel paper di riferimento
%                  -> GaussianStandard rumore gaussiano standard matlab
%                  -> GaussianManual rumore gaussiano implementato manualmente con intensità percDistruzione/1000
%                  -> SaltnPepper rumore sale e pepe con intensità percDistruzione/1000
tipoRumore = "Standard";

% Percentuale di distruzione dell'input utilizzata
v=25;

% Inizializzazione matrici di immagini ed etichette del dataset MNIST
[matImmagini, matEtichette] = caricaDataset('./risorse/train-images-idx3-ubyte', './risorse/train-labels-idx1-ubyte');

% Gli elementi che costituiscono training set, validation set e test 
% set vengono selezionati casualmente in modo da diversificare i test 
% effettuati con gli stessi input.
% Verranno a tal fine effettuate delle chiamate alla funzione rand() 
% ottenendo dei numeri casuali da 1 a 60000 (dimensione MNIST).
% Per evitare di considerare più volte una stessa immagine, si tiene 
% traccia degli indici già "pescati" attraverso una struttura dati.
% Il numero di elementi presi per ogni cifra deve essere uguale per 
% tutte le cifre. 
% Essendonci nel dataset 10 cifre diverse il numero di elementi preso 
% per ognuna deve essere uguale ad un decimo della dimensione del set 
% indicata in input. (Es. dim. training set=1000 -> ogni cifra 100
% elementi). Se viene inserita una dimensione che non sia un multiplo di
% dieci, questa viene arrotondata al valore multiplo di dieci che la 
% precede. 

% Vengono aggiornate le dimensioni dei set in modo da poter avere dieci
% partizioni della stessa dimensione. 
dimensioneTrainingSet = floor(dimensioneTrainingSet/10)*10;
dimensioneTestSet = floor(dimensioneTestSet/10)*10;
dimensioneValidationSet = floor(dimensioneValidationSet/10)*10;

indiceElemUtilizzati=zeros(1,60000);

% Creazione training set
[trainingSetImg,trainingSetLab,indiceElemUtilizzati]=creaSet(dimensioneTrainingSet,indiceElemUtilizzati, matEtichette,matImmagini);
% Creazione validation set
[validationSetImg,validationSetLab,indiceElemUtilizzati]=creaSet(dimensioneValidationSet,indiceElemUtilizzati, matEtichette,matImmagini);
% Creazione test set
[testSetImg,testSetLab,indiceElemUtilizzati]=creaSet(dimensioneTestSet,indiceElemUtilizzati, matEtichette,matImmagini);

% Creazione e inizializzazione dei livelli hidden
strutturaLivelliHidden(1)=struct('dimLivello',Class_nodiHidden,'funDiAttivazione',Class_funNodiHidden);

% Creazione rete neurale feed-forward multi-layer per la classificazione
reteNeuraleClass = creaReteFFML(784,10,Class_funNodiOutput,strutturaLivelliHidden,ClassminPeso,ClassmaxPeso,ClasslvHidden);

% Addestramento rete neurale
fprintf("\n Inizio addestramento della rete di classificazione\n");
pause(2);
[reteNeuraleClass, arrayErroriTS, arrayErroriVS] = addestraRete(reteNeuraleClass,Class_epoche,tipoAddestramento,trainingSetImg,validationSetImg,trainingSetLab,validationSetLab,Class_funErrore,Class_eta,flagSoftmax,false);

% Propagazione in avanti utilizzando come input il test set
[reteNeuraleClass] = forwardProp(reteNeuraleClass, testSetImg, true);

% Viene calcolata l'accuratezza della valutazione della rete rispetto al
% test set
[accuratezzaClassPura] = valutazioneRete(reteNeuraleClass.z{reteNeuraleClass.numLivelliHidden+1}, testSetLab);

% Funzione che stampa a video il confronto tra le label del test set e i
% risultati dati sulle stesse immagini dalla rete, il numero di immagini
% selezionate è dato dall'ultimo parametro
riscontroVisivo(testSetImg,reteNeuraleClass.z{reteNeuraleClass.numLivelliHidden+1},testSetLab,25);
pause(3);
fprintf("\n Fase addestramento autoencoder ");

% Sporco training e validation set per addestrare l'autoencoder
trainingSetSporcato = sporcaInput(trainingSetImg,v,tipoRumore);
validationSetSporcato = sporcaInput(validationSetImg,v,tipoRumore);
testSetSporcato = sporcaInput(testSetImg,v,tipoRumore);

% Costruzione e addestramento dell'autoencoder
[denoisingAutoEncoder,AE2LVarrayErroriTS,AE2LVarrayErroriVS]=addestraAutoEncoder(trainingSetSporcato,validationSetSporcato,trainingSetImg,validationSetImg,testSetSporcato);

pause(2);
fprintf("\n Fine addestramento autoencoder ");

% Test rete senza autoencoder su input sporcato
fprintf("\n Classificazione input sporcato senza autoencoder");

% Test senza autoencoder
reteNeuraleClassTest = forwardProp(reteNeuraleClass, testSetSporcato,true);
% Calcolo accuratezza senza autoencoder
[accuratezzaClassSenzaDenoising] = valutazioneRete(reteNeuraleClassTest.z{reteNeuraleClassTest.numLivelliHidden+1}, testSetLab);

% Stampa i dati utilizzati per l'addestramento della rete e
% l'accuratezza della rete rispetto al test set.
fprintf("\n Dimensione Training Set : %d\n Dimensione Validation Set : %d", dimensioneTrainingSet,dimensioneValidationSet);
fprintf("\n Struttura Rete di classificazione");
fprintf("\n Dimensione Test Set: %d\n Nodi hidden per livello: %d",dimensioneTestSet, Class_nodiHidden);
fprintf("\n eta : %f\n Numero di livelli hidden: %d",Class_eta ,ClasslvHidden);
fprintf("\n tipo addestramento: %s\n Numero di epoche: %d",tipoAddestramento ,Class_epoche);
fprintf("\n Accuratezza classificazione pura: %d%%\n ", int16(accuratezzaClassPura*100));

fprintf("\n Intensita rumore: %d%%",v);
fprintf("\n Tipo di rumore utilizzato: %s", tipoRumore);
fprintf("\n Risultato classificazione input sporcato senza autoencoder ");
fprintf("\n Accuratezza: %d%%\n ", int16(accuratezzaClassSenzaDenoising*100));

% L'input sporcato viene passato all'autoencoder e poi alla rete di
% classificazione
denoisingAutoEncoder = forwardProp(denoisingAutoEncoder, testSetSporcato, false);

% Classificazione dell'input ricostruito tramite denoising
reteNeuraleConDenoising = forwardProp(reteNeuraleClass, denoisingAutoEncoder.z{denoisingAutoEncoder.numLivelliHidden+1},true);

% Valutazione classificazione della reteNeuraleConDenoising
[accuratezzaClassConDenoising] = valutazioneRete(reteNeuraleConDenoising.z{reteNeuraleConDenoising.numLivelliHidden+1}, testSetLab);

% Funzione che stampa a video il confronto tra le label del test set e i
% risultati dati sulle stesse immagini dalla rete, il numero di immagini
% selezionate è dato dall'ultimo parametro
riscontroVisivo(testSetSporcato,reteNeuraleConDenoising.z{reteNeuraleConDenoising.numLivelliHidden+1},testSetLab,10);

% Stampa i dati utilizzati per l'addestramento della rete e
% l'accuratezza della rete rispetto al test set.
fprintf("\n Risultato finale classificazione rete con autoencoder ");
fprintf("\n Accuratezza : %d%%\n ", int16(accuratezzaClassConDenoising*100));
fprintf("\n Tempo impiegato per addestramento e test della rete: %f secondi \n",toc);