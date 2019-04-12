function [autoEncoder2LV,AE2LVarrayErroriTS,AE2LVarrayErroriVS] = addestraAutoEncoder(trainingSetSporcato,validationSetSporcato,trainingSetImg,validationSetImg,testSetSporcato)
% Utilizzato come script per implementare la struttura dell'autoencoder.
% Input:
% - trainingSetSporcato, immagini di input a cui è stato applicato del
% rumore;
% - validationSetSporcato, immagini usate come validation set;
% - trainingSetImg, immagini di input senza rumore, utilizzate come target;
% - validationSetImg, immgini del validation set senza rumore, utilizzate
% come target;
% - testSetSporcato, test set di immagini.
% Output:
% - autoEncoder2LV, la struttura autoeconder a due livelli
% - AE2LVarrayErroriTS, array errori della rete su training set;
% - AE2LVarrayErroriVS, array errori della rete su validation set;

% Configurazione iperparametri per i due autoencoder (un livello, due livelli)
AE1LVfunNodiHidden = @ReLU;
AE1LVfunNodiOutput = @funSigmoide;
AE2LVfunNodiHidden = @ReLU;
AE2LVfunNodiOutput = @funSigmoide;
funErrore = @funSommaQuadrati;
AE1LVnodiHidden = 128;
AE2LVnodiHidden = 32;
AE1LVepoche = 500;
AE2LVepoche = 500;
tipoAddestramento = "batch";
AE1LVminPeso = -0.09;
AE1LVmaxPeso = 0.09;
AE2LVminPeso = -0.09;
AE2LVmaxPeso = 0.09;
AE1LVlvHidden = 1;
AE2LVlvHidden = 3;
AE1LV_eta = 0.0002;
AE2LV_eta = 0.00002;
flagSoftmax = false;


% Creazione e inizializzazione dei livelli hidden del primo autoEncoder
strutturaLivelliHidden(1)=struct('dimLivello',AE1LVnodiHidden,'funDiAttivazione',AE1LVfunNodiHidden);

% Creazione rete neurale feed-forward multi-layer
autoEncoder1LV = creaReteFFML(784,784,AE1LVfunNodiOutput,strutturaLivelliHidden,AE1LVminPeso,AE1LVmaxPeso,AE1LVlvHidden);
autoEncoder1LV.W{2}=autoEncoder1LV.W{1}';

% Addestramento autoencoder AE1LV
[autoEncoder1LV, arrayErroriTS, arrayErroriVS] = addestraRete(autoEncoder1LV,AE1LVepoche,tipoAddestramento,trainingSetSporcato,validationSetSporcato,trainingSetImg,validationSetImg,funErrore,AE1LV_eta,flagSoftmax,false);

% Creo autoencoder a 2 lv
% Creazione e inizializzazione dei livelli hidden
AE2LVstrutturaLivelliHidden(1)=struct('dimLivello',AE1LVnodiHidden,'funDiAttivazione',AE2LVfunNodiHidden);
AE2LVstrutturaLivelliHidden(2)=struct('dimLivello',AE2LVnodiHidden,'funDiAttivazione',AE2LVfunNodiHidden);
AE2LVstrutturaLivelliHidden(3)=struct('dimLivello',AE1LVnodiHidden,'funDiAttivazione',AE2LVfunNodiHidden);

% Creazione rete neurale feed-forward multi-layer
autoEncoder2LV = creaReteFFML(784,784,AE2LVfunNodiOutput,AE2LVstrutturaLivelliHidden,AE2LVminPeso,AE2LVmaxPeso,AE2LVlvHidden);
% Recupero pesi autoencoder 1 lv
autoEncoder2LV.W{1}=autoEncoder1LV.W{1};
% Pesi legati 
autoEncoder2LV.W{3}=autoEncoder2LV.W{2}';
autoEncoder2LV.W{4}=autoEncoder2LV.W{1}';
% Addestro autoencoder 2LV
[autoEncoder2LV, AE2LVarrayErroriTS, AE2LVarrayErroriVS] = addestraRete(autoEncoder2LV,AE2LVepoche,tipoAddestramento,trainingSetSporcato,validationSetSporcato,trainingSetImg,validationSetImg,funErrore,AE2LV_eta,flagSoftmax,true);

% Vengono creati i grafici che mostrano l'andamento della funzione di errore
% rispetto al training e validation set dell'autoencoder a 2LV
figure('Name','Grafico errore AE2LV');
plot(AE2LVarrayErroriTS);
hold
plot(AE2LVarrayErroriVS);
legend('Errore rispetto al training set', 'Errore rispetto al validation set');

% Pesi legati
autoEncoder2LVtest=forwardProp(autoEncoder2LV, testSetSporcato, true);

% Stampa immagini
figure('units','normalized','outerposition',[0 0 1 1],'Name','Input - output denAutoencoder');
colormap(gray);

for i=1:10
    subplot(4,10,i+10)
	digit = reshape(autoEncoder2LVtest.X(i,:), [28,28]);
    imagesc(digit)
    axis off
    subplot(4,10,i+20)
    digit2 = reshape(autoEncoder2LVtest.z{autoEncoder2LV.numLivelliHidden+1}(i,:), [28,28]);
    imagesc(digit2)
    axis off
end

fprintf("\n struttura Denoising Autoencoder\n ");
fprintf("\n struttura nodi hidden : %d - %d - %d -%d -%d",784,AE1LVnodiHidden,AE2LVnodiHidden,AE1LVnodiHidden,784)
fprintf("\n eta : %f\n Numero di livelli hidden : %d",AE2LV_eta ,AE2LVlvHidden);
fprintf("\n tipo addestramento : %s\n Numero di epoche : %d\n",tipoAddestramento ,AE2LVepoche);