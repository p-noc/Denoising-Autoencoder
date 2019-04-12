function setImgTilde = sporcaInput(setImgInput,percDistruzione,tipoRumore)
%sporcaInput
% Aggiunge rumore alle immagini passate. 
% Prende in input:
% - setImgInput, insieme delle immagini da sporcare
% - percDistruzione, percentuale di distruzione delle immagini
% - tipoRumore : permette di selezionare che tipo di rumore applicare ,
% sono supportati -> "Standard" rumore utilizzato nel paper di riferimento
%                 -> "GaussianStandard" rumore gaussiano standard matlab, media 0 e stdev  0.01
%                 -> "GaussianManual" rumore gaussiano implementato manualmente con intensità percDistruzione/1000
%                 -> "SaltnPepper" rumore sale e pepe con intensità percDistruzione/1000
% Ritorna l'insieme delle immagini sporcate

setImgTilde=setImgInput;
% Estrae il numero di immagini da sporcare
numImmagini=size(setImgInput,1);
% In proporzione alla percentuale di distruzione data in input calcola
% quanti pixel sia necessario sporcare per ogni immagine
numPxDaSporcare=floor((percDistruzione*784)/100);
% Switch che gestisce i diversi tipi di rumore supportati
switch tipoRumore
    case 'Standard'
    % Per ogni immagine genera un numero di 0 uguale alla percentuale di
    % distruzione passata (percDistruzione)
        for i=1:numImmagini
            % Array contentente indice pixel sporcati
            arrayPxSporcati=zeros(1,784);
            % Contatore pixel sporcati
            numPxSporcati=0;
            while numPxSporcati<numPxDaSporcare
                % Scelgo casualmente quale pixel sporcare
                randPx=floor((784-1).*rand(1)+1);
                if arrayPxSporcati(randPx)==0
                    setImgTilde(i,randPx)=0;
                    arrayPxSporcati(randPx)=1;
                    numPxSporcati=numPxSporcati+1;
                end
            end
        end
    case 'GaussianStandard'
    %"GaussianStandard" rumore gaussiano standard matlab, media 0 e stdev  0.01
        for i=1:numImmagini
            digit = reshape(setImgInput(i,:), [28,28]);
            digitTilde = imnoise(digit,'gaussian');
            setImgTilde(i,:) = reshape(digitTilde,[784,1]);     
        end


    case 'GaussianManual'
    %"GaussianManual" rumore gaussiano implementato manualmente con intensità percDistruzione/1000
        for i=1:numImmagini
            digit = reshape(setImgInput(i,:), [28,28]);
            digitTilde = double(digit) + (percDistruzione/1000)*randn(size(digit));
            setImgTilde(i,:) = reshape(digitTilde,[784,1]);     
        end

    case 'SaltnPepper'
    %"SaltnPepper" rumore sale e pepe con intensità percDistruzione/1000
        for i=1:numImmagini
            digit = reshape(setImgInput(i,:), [28,28]);
            digitTilde = imnoise(digit,'salt & pepper', percDistruzione/1000);
            setImgTilde(i,:) = reshape(digitTilde,[784,1]);     
        end
    otherwise
    fprintf("\nIl tipo di rumore richiesto non è supportato, non verrà applicato rumore al dataset");
end

end

