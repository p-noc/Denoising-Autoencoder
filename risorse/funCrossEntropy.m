function z = funCrossEntropy(x,y,daDerivare)
% funCrossEntropy:
% Una delle possibili funzioni utilizzate per il
% calcolo dell'errore. Dati due valori x e y viene 
% calcolato il valore della funzione cross entropy
% rispetto ai valori di input.
% Se richiesto, viene calcolata e restituita la derivata
% della funzione.

% Viene richiesta la derivata della cross entropy
    if exist('daDerivare','var')
            z=x-y;
    % Viene la calcolata la cross entropy senza derivata
    else
        % Distinguiamo il calcolo per i valori maggiori
        % o uguali di 0. Quelli uguali a 0 danno probemi
        % per il calcolo del logaritmo.
        
        % Valori >0:
        y(x>0)=y(x>0) .* log(x(x>0));
        
        % Valori =0;
        y(x==0)=y(x==0)*(-708); %Logaritmo del valore più 
                                %piccolo rappresentabile.
        z=-sum(y);
    end
end

