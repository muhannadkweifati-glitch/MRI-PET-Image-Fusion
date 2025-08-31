
%% **1. Load MRI and PET Images**
%% Select MRI image
[mriFile, mriPath] = uigetfile({'*.png;*.jpg;*.tif','Image Files'}, 'Select MRI Image');
if isequal(mriFile,0)
    error('No MRI image selected!');
end
Imr = im2double(imread(fullfile(mriPath, mriFile)));

%% Select PET image
[petFile, petPath] = uigetfile({'*.png;*.jpg;*.tif','Image Files'}, 'Select PET Image');
if isequal(petFile,0)
    error('No PET image selected!');
end
Ipe = im2double(imread(fullfile(petPath, petFile)));


% Convert MRI to Grayscale (If Not Already)
if size(Imr,3) == 3
    Imr = rgb2gray(Imr);
end

% Convert PET Image to YCbCr Format
IpeYCbCr = rgb2ycbcr(Ipe);
Y_pet = IpeYCbCr(:,:,1);  % Extract Luminance (Y) Channel
Cb_pet = IpeYCbCr(:,:,2); % Chrominance Blue (Cb)
Cr_pet = IpeYCbCr(:,:,3); % Chrominance Red (Cr)

% Resize images to match dimensions
[m, n] = size(Imr);
Y_pet = imresize(Y_pet, [m, n]);  

%% **2. Multi-Scale Decomposition (Low-Pass & High-Pass Filtering)**
sigma = 1;  % Standard deviation for Gaussian filter
Imr_LF = imgaussfilt(Imr, sigma);  % Low-Frequency Component (MRI)
PET_LF = imgaussfilt(Y_pet, sigma);  % Low-Frequency Component (PET)

% High-Frequency Components (Subtract Low-Frequency from Original)
Imr_HF = Imr - Imr_LF;
PET_HF = Y_pet - PET_LF;
PET_HF = PET_HF * (mean(abs(Imr_HF(:))) / mean(abs(PET_HF(:))));

%% **3. Multi-Directional Filtering (Gabor)**
theta = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5];  % More Directions
lambda = 4;  % Wavelength
gaborArray = gabor(lambda, theta);  % Create Gabor Filters

% Preallocate space for directional feature extraction
MRI_Dir = zeros(size(Imr_HF,1), size(Imr_HF,2), length(theta));
PET_Dir = zeros(size(PET_HF,1), size(PET_HF,2), length(theta));

% Apply Gabor Filters
for i = 1:length(gaborArray)
    MRI_Dir(:,:,i) = imgaborfilt(Imr_HF, gaborArray(i).Wavelength, gaborArray(i).Orientation);
    PET_Dir(:,:,i) = imgaborfilt(PET_HF, gaborArray(i).Wavelength, gaborArray(i).Orientation);
end

%% **4. Fusion of Low-Frequency and High-Frequency Components**
% Compute adaptive weights based on variance
w1 = 50;
w2 = 50;

% Low-Frequency Fusion (Adaptive Weighting)
LF_Fused = w1 * Imr_LF + w2 * PET_LF;

% High-Frequency Fusion using Maximum Fusion Rule
for i = 1:length(gaborArray)
    HF_Fused(:,:,i) = max(abs(MRI_Dir(:,:,i)), abs(PET_Dir(:,:,i)));
end

%% **5. Inverse Transformation (Reconstruction)**
% Sum Fused High-Frequency Components Across All Directions
HF_Reconstructed = sum(HF_Fused, 3);  

% Reconstruct the Fused Intensity (Y Channel)
Fused_Y = LF_Fused + HF_Reconstructed;

%% **6. Post-Processing (Brightness & Contrast Fixes)**
% Apply Adaptive Histogram Equalization (Enhances Contrast)

% Adjust Brightness (Fix ALI)

% Apply 

% Normalize the Fused Y Channel
Fused_Y = mat2gray(Fused_Y);

%% **7. Convert Back to RGB (Using YCbCr)**
Fused_YCbCr = cat(3, Fused_Y, Cb_pet, Cr_pet);
Fused_RGB = ycbcr2rgb(Fused_YCbCr);

%% **8. Display Results**
figure; 
%subplot(1,3,1), imshow(Imr, []), title('MRI Image (Grayscale)');
%subplot(1,3,2), imshow(Ipe, []), title('Original PET Image (RGB)');
%subplot(1,3,3), imshow(Fused_RGB), title('Final Fused Image (RGB)');
imshow(Fused_RGB), title('GBAOR PA');

%% **9. Compute Image Fusion Performance Metrics**
Fused_Y_gray = im2gray(Fused_Y);
Y_mri_gray = im2gray(Imr);

% Compute Performance Metrics
entropy_fused = entropy(Fused_Y_gray);
ALI = mean(Fused_Y_gray(:));
CI = std(Fused_Y_gray(:));
sharpness = sum(sum(abs(imgradient(Fused_Y_gray))));

edge_mri = edge(Y_mri_gray, 'Canny');
edge_fused = edge(Fused_Y_gray, 'Canny');
QABF = sum(sum(edge_fused & edge_mri)) / sum(sum(edge_mri));

% Display Metrics
fprintf('\nFinal Fusion Performance Metrics:\n');
fprintf('Entropy: %.4f\n', entropy_fused);
fprintf('ALI: %.4f\n', ALI);
fprintf('CI: %.4f\n', CI);
fprintf('Sharpness: %.4f\n', sharpness);
fprintf('QAB/F: %.4f\n', QABF);

% Save Fused Image
