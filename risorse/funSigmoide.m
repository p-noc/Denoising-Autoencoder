function z = funSigmoide(x,daDerivare)
% funSigmoide:
% Prende in input un valore x e restitusce il risultato  
% della funzione sigmoide nel caso in cui la flag per la 
% derivata sia inattiva, il valore derivato altrimenti.

    % Viene richiesta la derivata 
    if exist('daDerivare','var')
        z=funSigmoide(x).*(1-funSigmoide(x));
    else % Viene calcolato il valore della sigmoide
        z=1.0 ./ (1.0+exp(-x));
    end
end

