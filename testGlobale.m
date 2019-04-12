% Script utilizzato per la sperimentazione orientata al calcolo della media
% dell'accuretezza e della deviazione standard della rete di
% classificazione.

% Inizializzazione variabili
clearvars;
numeroRipetizioni = 5;
M = zeros(8,6);
trainingSet = 1000;
validationSet = 250;
testSet = 250;

%senza rumore, standard 0
row=1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "Standard", 0);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);

%standard 25
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "Standard", 25);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);

%standard 50
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "Standard", 50);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);


%Standard 75
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "Standard", 75);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);


%SaltnPepper 25
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "SaltnPepper", 25);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);

%SaltnPepper 50
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "SaltnPepper", 50);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);


%SaltnPepper 75
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "SaltnPepper", 75);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);


%GaussianStandard
row=row+1;
tempAccNorm = zeros([1,numeroRipetizioni]);
tempAccSporc = zeros([1,numeroRipetizioni]);
tempAccRicostruito = zeros([1,numeroRipetizioni]);
for i = 1:numeroRipetizioni
[AccNorm,AccSporc,AccRicostruito] = testSingolaConfig(trainingSet, validationSet, testSet, "GaussianStandard", 0);
        tempAccNorm(i) = AccNorm;
        tempAccSporc(i) = AccSporc;
        tempAccRicostruito(i) = AccRicostruito;
end        
M(row,1) = mean(tempAccNorm);
M(row,2) = std(tempAccNorm);
M(row,3) = mean(tempAccSporc);
M(row,4) = std(tempAccSporc);
M(row,5) = mean(tempAccRicostruito);
M(row,6) = std(tempAccRicostruito);

csvwrite('AccuratezzeMedie.txt',M);

fprintf("\n FINE\n");
