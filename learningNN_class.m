
function [E, E_val, W1, W2, ct] = learningNN_class( Xtrain, Ytrain, Xval, Yval, hidden_units, alpha, max_iter )
tic

output = 1;  % número de unidades de salida

% Inicialización aleatoria de los pesos en cada capa
W1 = 5 * randn(hidden_units, size(Xtrain, 1)); % pesos de la capa 1
W2 = 5 * randn(output, hidden_units + 1); % pesos de la capa 2

E = zeros(1, max_iter); % inicializar la variable para almacenar el error de entrenamiento
E_val = zeros(1, max_iter); % inicializar la variable para almacenar el error de validación

num_pattern = size(Xtrain, 2);

for i = 1:max_iter
    incr1 = zeros(size(W1));
    incr2 = zeros(size(W2));
    
    % Desordenar los patrones de manera aleatoria
    idx = randperm(num_pattern);
    Xtrain_shuffled = Xtrain(:, idx);
    Ytrain_shuffled = Ytrain(idx);

    for p = 1:num_pattern  
        [DW1, DW2] = gradientNN_class(W1, W2, Xtrain_shuffled(:,p), Ytrain_shuffled(p));
        
        % Actualizar los pesos para cada patrón
        W1 = W1 - alpha * DW1;
        W2 = W2 - alpha * DW2;
        
    end
    
    % calcula el error para trainset y testset:
    E(i) = GetError_class(W1, W2, Xtrain, Ytrain);
    E_val(i) = GetError_class(W1, W2, Xval, Yval);
end

ct = toc;

end