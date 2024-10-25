function gabor_template(kern_size, lambda, phi, sigma, gamma, orientations)
 
IM = imread('images/gardel.jpeg');  % image name
IM = sum(IM,3);
image_size = size(IM);

figure(1),
imagesc(IM), title('Imagen original')  % show the image
colormap('gray'), axis image
axis off

%% Parameters of the filter
kern_size = kern_size; %50 dimensión del kernel
lambda = lambda; %0; % longitud de onda% frecuencia
phi = phi; %0; % cambio fase cos
sigma = sigma; %0.5; % controla el radio del filtro% qué tan ancho o estrecho es la campana gaussiana
gamma = gamma; %0.3; % controla la forma del filtro (aplanado)% qué tan ancho o estrecho es la campana gaussiana
orientations = orientations; % de ángulos en radianes %selección 4 ángulos. Por ejemplo [0 pi/4 pi/2 3/4*pi]


%% Aplicar gabor para cada orientación

for i = 1:4  % angles loop (use 4 values for the angle; if you want to use more you have to change the code for plotting)
 
    kernel = gabor_kernel(kern_size, orientations(i), lambda, phi, sigma, gamma, true);  % funcion que devuelve el kernel, el parametro fig al final se puede dejar en false para no tener tantas figuras para los angulos del kernel
    filt_img = conv2(double(IM),kernel, 'same');  % image convolution with the kernel
    
    % plots of the result of the convolution
    figure;
    imagesc(filt_img)  
    colormap('gray'), axis image
    title(['angle = ' num2str(orientations(i))])  % write here the value of the angle
    axis off
    colorbar
end
end