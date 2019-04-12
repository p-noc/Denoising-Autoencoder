function riscontroVisivo(imgCifre,outputRete,labelCorrette,numImmagini)
% Stampa le immagini contenente le cifre accompagnate dall'output del
% relativo output della rete e la label corretta dell'immagine.

rispostaClasse=adattaRisposta(outputRete);
figure('units','normalized','outerposition',[0 0 1 1],'Name','Risultati classificazione');
colormap(gray);
for i=1:numImmagini
    j=1;
    z=1;
    while rispostaClasse(i,j)~=1
        j=j+1;
    end
    while labelCorrette(i,z)~=1
        z=z+1;
    end

    subplot(ceil(sqrt(numImmagini)),ceil(sqrt(numImmagini)),i)
	digit = reshape(imgCifre(i,:), [28,28]);
	imagesc(digit)
    title(sprintf("Out rete: %d / Label: %d", int16(j-1),int16(z-1)),'FontSize',8)
    axis off
end


end

