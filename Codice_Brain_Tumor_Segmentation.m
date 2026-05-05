%% Progetto: Brain Tumor Segmentation con Watershed Segmentation
% ---- Russo Andrea ---- %

clear; clc; close all force;

%% CARICAMENTO DATI 

imagePath = "./Task01_BrainTumour/imagesTr/BRATS_009.nii.gz"; 
labelPath = "./Task01_BrainTumour/labelsTr/BRATS_009.nii.gz";

fullVolume = niftiread(imagePath); % Carica volume 4D
groundTruth = niftiread(labelPath); % Carica label 3D



% STAMPA PARAMETRI FULLVOLUME
[x, y, z] = size(fullVolume);
z_volume = z/4;
fetta = round(z_volume/2);       % scelta fetta centrale

fprintf("========= PARAMETRI FULLVOLUME =========\n");
fprintf("Grandezza di x:        %d\n", x);
fprintf("Grandezza di y:        %d\n", y);
fprintf("Numero di fette (z):   %d\n", z_volume);
fprintf("----------------------------\n");
fprintf("Fetta in mezzo:        %d\n", fetta);
fprintf("========================================\n");


% STAMPA PARAMETRI GROUNDTRUTH
[x, y, z] = size(groundTruth);

fetta_label = round(z/2);       % scelta fetta centrale

fprintf("\n\n\n========= PARAMETRI GROUNDTRUTH =========\n");
fprintf("Grandezza di x:        %d\n", x);
fprintf("Grandezza di y:        %d\n", y);
fprintf("Numero di fette (z):   %d\n", z);
fprintf("----------------------------\n");
fprintf("Fetta in mezzo:        %d\n", fetta_label);
fprintf("=========================================\n");




%% VISUALIZZAZIONE VOLUMETRICA

A = mat2gray(fullVolume(:,:,fetta, 1)); % NORMALIZZAZIONE SINGOLA FETTA
V_norm = mat2gray(fullVolume(:,:,:,1)); % Normalizzazione Tutto

% VISUALIZZAZIONE 2D: Panoramica di tutte le fette (Volume)
figure("Name", "Panoramica Fette MRI (FLAIR)", "NumberTitle", "off");
montage(V_norm);     % uso di V_norm
colormap(gray(256));
title("Volume FLAIR - 155 fette");

%----------------------------------------------------------------

Label_A = mat2gray(groundTruth(:,:,fetta_label)); % NORMALIZZAZIONE LABEL
V_label_norm = mat2gray(groundTruth);

% VISUALIZZAZIONE 2D: Panoramica di tutte le fette (Label)
figure("Name", "Panoramica Ground Truth", "NumberTitle", "off");
montage(V_label_norm);
colormap(gray(256));
title("Volume Ground Truth - 155 fette");





%% PRE-PROCESSING

% Applicazione di un Avarage Filter 

A2 = A;
k1 = 3;  % quindi avrò 1/9 per fare la media,

average_filter = fspecial("average", k1);   %creo box filter

A2_filtered = imfilter(A2, average_filter);     % applico filtro

figure('Name', "Confronto");

subplot(1,2,1);
imshow(A); axis image; title("Prima del Preprocessing");

subplot(1,2,2);
imshow(A2_filtered); axis image; 
title("Dopo il Preprocessing con Average Filter");


A = A2_filtered;





%% SKULL STRIPPING


figure;

subplot(2,3,1);     % immagine iniziale
imshow(A, []); axis on; title("Immagine con Filtro");

%-------------------------------------%

subplot(2,3,2:3);

% Invece di 11 (su 255), usiamo circa 0.04 (che è 11/255)
soglia_fondo = 11/255;


imhist(A(A>=soglia_fondo))  % realizzo istogramma
hold on;
level = graythresh(A);                  % stabilisco il threshold 
fprintf("\n\n=========================================");
fprintf("\nIl valore soglia 'level' è: %d\n", level);
fprintf("=========================================\n");
stem(level, 500, "r", "LineWidth", 2);      % disegno la linea rossa
hold off;

binaryImage = imbinarize(A, level);

subplot(2,3,4) % binary image
imshow(binaryImage, []); axis on; title("Binary Image Thresholded");

%-------------------------------------%

binaryImage = bwareafilt(binaryImage, 2);   % estraggo i due oggetti       
                                            % bianchi più grandi

binaryImage = imopen(binaryImage, true(5)); % per staccare gli oggetti  
                                            % che sono legati da un filo di
                                            % pixel e distrugge i piccoli
                                            % filamenti

binaryImage = bwareafilt(binaryImage, 1);    % estraggo il singolo oggetto 
                                             % più grande bianco (cervello)
                                             % dopo aver pulito con
                                             % l'apertura

binaryImage = imfill(binaryImage, "holes");    % riempio i buchi

binaryImage = imdilate(binaryImage, true(5));  % dilatazione della maschera

subplot(2,3,5) % binary image
imshow(binaryImage, []); axis on; title("Binary Image Finale");

%-----------------------------------%

skullFreeImage = A;
skullFreeImage(~binaryImage) = 0;

subplot(2,3,6) % Immagine finale ritagliata
imshow(skullFreeImage, []); axis on; title("Immagine con binaryImage");


A = skullFreeImage;



%% UTILIZZO DI SOBEL

M = imgradient(A);  % Gradient Magnitude 

figure("Name", "Analisi del Gradiente di Sobel");

subplot(1,2,1); 
imshow(M, []); axis on; title("Gradient Magnitude (M)");

subplot(1,2,2); 
imagesc(M); colorbar; axis image; title("Mappa Altitudinale per Watershed");

A_Originale = A;




%% VISUALIZZAZIONE TECNICA WATERSHED

L = watershed(M);     % Problema di Oversegmentation


k = 0.135;
B = imhmin(M, k);     % Marker Control
L2 = watershed(B);    

figure("Name", "Confronto");

subplot(2,2,1);
imshow(M); axis image; title("Immagine A (Magnitude)");

subplot(2,2,2);
imshow(B); axis image; title("Immagine B dopo imhmin su A (Magnitude)");

subplot(2,2,3);
imagesc(L); axis image; title("Immagine L da A (Magintude)");

subplot(2,2,4);
imagesc(L2); axis image; title("Immagine L2 da B");




%% SOVRAPPOSIZIONE SEGMENTAZIONE

A1 = A_Originale;
[x, y] = size(A1); 
C1 = uint8(ones(x, y)*255); % Crea una matrice tutta BIANCA (valore 255)
C1(L == 0) = 0;     % realizzo delle linee nere

% sovrappongo 
A1(C1 == 0) = 0;    % non posso usare la sottrazione A - C perché vengono  
                    % sottratti dei pixel tra due scale diverse:
                    % A è un'immagine normalizzata (grazie a mat2gray), 
                    %   quindi i suoi pixel vanno da 0 a 1.
                    %
                    % C è una matrice uint8, quindi i suoi pixel vanno 
                    %   da 0 a 255

%-------------------------------

A2 = A_Originale;
[x, y] = size(A1); 
C2 = uint8(ones(x, y)*255); % Crea una matrice tutta BIANCA (valore 255)
C2(L2 == 0) = 0;     % realizzo delle linee nere

% sovrappongo 
A2(C2 == 0) = 0; 




figure("Name", "Confronto Sovrapposizione");
subplot(2,2,1);
imshow(C1); axis image; title("Maschera senza imhmin");

subplot(2,2,2);
imshow(C2); axis image; title(["Maschera con k = ", num2str(k)]);

subplot(2,2,3);
imshow(A1); axis image; title("Sovrapposizione senza imhmin");

subplot(2,2,4);
imshow(A2); axis image; title(["Sovrapposizione con k = ", num2str(k)]);




%% OVERLAY FINALE


Risultato = A_Originale;
Risultato(L2 == 0) = 1; % Disegna le linee (0) in bianco (1)


figure("Name", "Risultato Finale Sobel-Watershed");
imshow(Risultato, []);
title("Segmentazione con bordi di Sobel");





%% OPERAZIONE MORFOLOGICA

L2_maschera = L2 > 0;   % creo maschera binaria

% applicazione Erosione
B = strel("disk", 1);   % creo uno Structuring Element
L2_erosa = imerode(L2_maschera, B); 

L2_originale = L2;
L2_originale(~L2_erosa) = 0;


figure("Name", "Operazione Morfologica - Maschera L2 EROSA");
subplot(2,2,1);
imagesc(L2); axis image; title("L2 Originaria");

subplot(2,2,2);
imshow(L2_maschera); axis image; title("Maschera L2 binaria");

subplot(2,2,3);
imshow(L2_erosa); axis image; title("L2 Binaria EROSA");

subplot(2,2,4);
imagesc(L2_originale); axis image; title("L2 Originale EROSA");



% calcolo le proprietà di tutte le regioni Erose
stats = regionprops(L2_originale, A_Originale, "MeanIntensity", "PixelIdxList");
stats_totale = regionprops(L2, "PixelIdxList");


% trovo il valore dell'intensità massima tra tutte le regioni
allMeans = [stats.MeanIntensity];
maxIntensity = max(allMeans);

% recupero quelle regioni che hanno un valore vicino al max per include
% l'intera massa tumorale
k2 = 0.7; % prelevo il 70% del valore max

soglia_accettazione = maxIntensity * k2; 
indiciTumore = find(allMeans > soglia_accettazione);

% creo una maschera unendo tutte queste regioni
binaryTumor = false(size(A_Originale));

for i = 1:length(indiciTumore)
    binaryTumor(stats_totale(indiciTumore(i)).PixelIdxList) = 255;
end

figure("Name", "Operazione Morfologica - binaryTumor");
imshow(binaryTumor); title("binaryTumor");


% Overlay
figure("Name", "Rilevazione Tumore Completa");
imshow(A_Originale, []); hold on;
visboundaries(binaryTumor, "Color", "r", "LineWidth", 1.5); % creo bordi
title("Tumore intero rilevato (Core + Edema)");






%% CALCOLO DELL'AREA E POSIZIONE DEL TUMORE 

% Recuperiamo le informazioni spaziali dal file originale
info = niftiinfo(imagePath);
risoluzione = info.PixelDimensions; % risoluzione in mm 
                                   % [Horizontal, Vertical, SliceThickness]


% calcolo delle dimensioni fisiche H e V
H = risoluzione(1);     % dimensione orizzontale immagine
V = risoluzione(2);     % dimensione verticale immagine

% area di ogni singolo pixel
A_pixel = V * H; 

% conteggio dei pixel nel tumore
pixel_totali_tumore = sum(binaryTumor(:)); 

% calcolo dell'area totale del tumore
area_tumore = A_pixel * pixel_totali_tumore;



% posizione emisferi
[righe, colonne] = size(A_Originale);
meta_riga = round(righe / 2); % divido a metà 

% divido la maschera del tumore in parte Alta e parte Bassa
parte_superiore = binaryTumor(1:meta_riga, :); % emisfero destro 
parte_inferiore  = binaryTumor(meta_riga+1:end, :); % emisfero sinistro

% conto i pixel
pixel_su = sum(parte_superiore(:));
pixel_giu = sum(parte_inferiore(:));

if pixel_su == pixel_giu && pixel_su > 0
    posizione = "Centro del cervello";
elseif pixel_su > pixel_giu
    posizione = "Emisfero Destro";
else
    posizione = "Emisfero Sinistro";
end




%% STAMPA DEI RISULTATI
fprintf("\n\n\n============= CALCOLO TUMORE ==================\n");
fprintf("Risoluzione Orizzontale (H): %.2f mm\n", H);
fprintf("Risoluzione Verticale (V):   %.2f mm\n", V);
fprintf("Area di un singolo pixel:    %.2f mm^2\n", A_pixel);
fprintf("------------------------------------------\n");
fprintf("Numero Pixel Tumore:         %d pixel\n", pixel_totali_tumore);
fprintf("AREA TOTALE TUMORE:          %.2f mm^2\n", area_tumore);
fprintf("===============================================\n");
fprintf("Posizione Tumore:            %s\n", posizione);
fprintf("===============================================\n");


% visualizzazione posizione tumore
figure("Name", "Divisione Emisferi");
imshow(A_Originale); hold on;
title(["Area Stimata del Tumore: ", num2str(area_tumore, "%.2f"), " mm^2"]);
yline(meta_riga, 'y', 'LineWidth', 2); % linea orizzontale gialla
visboundaries(binaryTumor, "Color", "r", "LineWidth", 1.5); % bordi rossi

text(10, meta_riga - 20, 'Emisfero Destro', 'Color', 'y');
text(10, meta_riga + 20, 'Emisfero Sinistro', 'Color', 'y');


%--------------------------------------------------


A_Originale(~binaryTumor) = 0;

% visualizzazione finale con il calcolo dell'area
figure("Name", "Confronto Finale");
subplot(1,2,1);
imshow(A_Originale, []);
title(["Area Stimata del Tumore: ", num2str(area_tumore, "%.2f"), " mm^2"]);

subplot(1,2,2);
imshow(Label_A); axis image; title("Area Tumore effettivo");




%% METRICHE

fprintf("\n\n\n\n================== CALCOLO METRICHE =================\n");

% METRICHE GLOBALI
val_min = min(A_Originale(:));
val_max = max(A_Originale(:));
val_medio = mean(A_Originale(:));
val_varianza = var(A_Originale(:));

fprintf("\n--- Metriche Globali ---\n");
fprintf("Min: %.4f | Max: %.4f\n", val_min, val_max);
fprintf("Media: %.4f | Varianza: %.4f\n", val_medio, val_varianza);




% PROPRIETÀ DELLE REGIONI
stats_tumore = regionprops(binaryTumor, "Centroid"); 

fprintf("\n\n--- Proprietà delle Regioni ---\n");

if ~isempty(stats_tumore)
    centroide = stats_tumore(1).Centroid;  % calcolo del centroide

    fprintf("Centroide del Tumore Segmentato (X, Y):\n");
    fprintf("X: %.2f\n", centroide(1));
    fprintf("Y: %.2f\n", centroide(2));
else
    fprintf("Nessuna regione rilevata.\n");
end




% METRICHE DI ERRORE
% SSD tra Segmentazione e Ground Truth
GT_binario = groundTruth(:,:,fetta) > 0;  % trasformo il Label in un'img binaria
diff_sq = (double(binaryTumor) - double(GT_binario)).^2;
ssd = sum(diff_sq(:));

% confronti binari
figure("Name", "Confronto Binarizzazioni");
subplot(1,2,1);
imshow(binaryTumor, []);
title("binaryTumor Calcolato");

subplot(1,2,2);
imshow(GT_binario); axis image; 
title("Tumore effettivo - Binario");

pixel_effettivi_GT = sum(GT_binario(:));
error_rate = (ssd/pixel_effettivi_GT)*100;

fprintf("\n\n--- Metriche di Errore ---\n");
fprintf("Numero Pixel Tumore Calcolati:    %d pixel\n", pixel_totali_tumore);
fprintf("Numero Pixel Tumore GroundTruth:  %d pixel\n", pixel_effettivi_GT);
fprintf("SSD (Sum of Squared Differences): %.2f pixel \n", ssd);
fprintf("Error Rate: %.2f%% \n", error_rate);

fprintf("\n====================================================\n");


% ---- Russo Andrea ---- %


