# denoising-autoencoder

Dopo aver studiato l'articolo "Extracting and composing robust features with denoising autoencoders'' è stato utilizzato il linguaggio MATLAB per implementare le parti che vanno a comporre la creazione, la gestione e l'analisi dei risultati di una rete neurale multistrato feed-forward, al fine di replicare il lavoro descritto nel suddetto articolo.
In particolare, vanno a formare il progetto finale la creazione di una rete neurale per la classificazione degli elementi del dataset MNIST ed un "denoising autoencoder" per la ricostruzione delle immagini sottoposte a rumore. 
L'obbiettivo finale è quello di valutare l'accuratezza della rete per la classificazione in risposta a diversi tipi di input. Gli input forniti alla rete sono rappresentati dagli elementi originali del dataset MNIST \ref{parmnist}, dagli stessi elementi sottoposti a rumore e infine dagli elementi ricostruiti dall'autoencoder.
Il tipo di rumore utilizzato dagli autori dell'articolo è generato azzerando in maniera randomica un certo numero di pixel dell'immagine.
È stato inoltre deciso di valutare le prestazioni delle reti anche con altri tipi di rumore.

[a link](https://github.com/p-noc/denoising-autoencoder/blob/master/creaSet.m)
