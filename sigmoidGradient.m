function g = sigmoidGradient(z)
%   g = sigmoidGradient(z) calcula la derivada de la funcion sigmoide evaluada
%   en z. Tiene que funcionar independientemente de que z sea un vector o 
%   una matriz, deberia devolver la derivada para cada elemento. 

g = zeros(size(z));

% ====================== Inserta tu código aquí ======================

 g = sigmoid(z) .* (1 - sigmoid(z));

% =============================================================

end
