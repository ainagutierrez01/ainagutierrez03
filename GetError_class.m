function E = GetError_class(W1,W2,x,y)
    
num_pattern = size(x, 2);
e = zeros(1,num_pattern);

for p = 1:num_pattern  % pattern loop       
    xp = x(:,p); % entrada para la muestra p
    yp = y(p);   % target para la muestra p
    
    % calcula activación para las neuronas ocultas:
    z_h = W1*xp;  % suma antes de la sigmoide
    a_h = sigmoid(z_h);  % activación de las neuronas ocultas
    a_h = [ones(size(yp)); a_h];  % agrega la neurona cuyo valor es siempre igual a uno y que sirve de sesgo a la siguiente capa

    % calcula activación de la neurona de salida:
    z_out = W2*a_h;  % suma antes de la función de respuesta de la neurona de salida
    a_out = sigmoid(z_out);  % calcula respuesta neurona de salida (regresión: identidad -- clasificación: sigmoide)

    % Calculamos el error de entropía cruzada para esta muestra:
    e(p) = -mean (yp .* log(a_out) + (1 - yp) .* log(1 - a_out)); % Entropía cruzada

end  
    
    
E = mean(e);
