function [immaginiCifre,etichetteCifre] = caricaDataset(percorsoImmagini,percorsoEtichette)
% caricaDataset:
% Funzione utilizzata per caricare il dataset MNIST.
% Il dataset è formato da 60000 immagini che rappresentano
% le cifre da 0 a 9 scritte a mano. Ogni immagine è rappresentata
% attraverso una matrice di pixel 28x28 (784px) in cui 
% ogni cella rappresenta l'intensità del singolo pixel
% in una rappresentazione a scala di grigi (quindi da 0 a 255).
% Ad ogni immagine è associata un'etichetta che indica la cifra
% rappresentata dall'immagine. 
% Questa funzione utilizza le funzioni realizzate
% dall'università di Stanford e permettono di ricavare due 
% matrici che contengono immagini e labels.

% La matrice immaginiCifre è una matrice 60000x784, nella quale
% ogni colonna rappresenta un'immagine, e le righe i valori di 
% intensità associati ai pixel. 
immaginiCifre = (loadMNISTImages(percorsoImmagini))';

% La matrice etichetteCifre è una matrice 60000x1 in cui ogni riga i
% contiene la cifra che corrisponde all'i-esima immagine. 
etichetteCifre = loadMNISTLabels(percorsoEtichette);
end

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
