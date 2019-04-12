function z = funSommaQuadrati(x,y,daDerivare)
%funSommaQuadrati
% Dati due valori x e y calcola la somma dei quadrati.
% Se richiesto ritorna la derivata di questa funzione.

    % Viene richiesta la derivata
    if (exist('daDerivare','var'))
        z=x-y;
    else
        z=0.5*sum(((x-y).^2));
    end
end

