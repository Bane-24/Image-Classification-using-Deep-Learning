# Bird Species Classification: Handcrafted Features vs. Deep Learning


## Overview

This project explores and compares **handcrafted feature extraction methods** (HOG + SVM) and **deep learning-based approaches** (e.g., ResNet50) for bird species classification. The study leverages the **Caltech-UCSD Birds 200 (CUB-200-2011)** dataset, utilizing both bounding box and full image inputs.

---

## Key Features

- **Handcrafted Methods**: Feature extraction with HOG and SVM classifiers.
- **Deep Learning Approaches**: CNNs and pre-trained **ResNet50** for classification.
- **Data Augmentation**: Applied during training for enhanced generalization.
- **Evaluation**: Fivefold cross-validation for robust performance metrics.

---

## Tech Stack

- **Dataset**: [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- **Handcrafted Features**: HOG, Color Histogram
- **Deep Learning**: ResNet50, CNN
- **Frameworks**: MATLAB Deep Learning Toolbox
- **Libraries**: `augmentedImageDatastore`, `imagedataaugmenter`

---

## Methodology

### Handcrafted Features
1. Extract features using HOG and Color Histograms.
2. Train Support Vector Machines (SVM) with hyperparameter tuning for classification.

### Deep Learning
1. Use ResNet50 and custom CNN architectures.
2. Train models with and without bounding box input.
3. Apply data augmentation techniques (rotation, scaling, translations).

### Experiments
1. **Handcrafted Features + SVM**: Full image vs. bounding box input.
2. **CNN**: Multiple configurations tested on full image and bounding box input.
3. **ResNet50**: Pre-trained model fine-tuned for classification.

---

## Results
A Sample result of a CNN is uploaded as a PDF in the repo.

| Method            | Input          | Accuracy (%) | Notes                                   |
|--------------------|----------------|--------------|-----------------------------------------|
| HOG + SVM         | Full Image     | 52%          | Basic handcrafted features.             |
| ResNet50          | Bounding Box   | 89%          | Best-performing approach.               |
| CNN               | Bounding Box   | 70%          | Enhanced by deeper architectures.       |

---

## Sample Code: ResNet50 Training in MATLAB

```matlab
% Load dataset with bounding box
data = augmentedImageDatastore([224, 224], trainImgs, 'DataAugmentation', imageAugmenter);

% Load ResNet50 model
net = resnet50;
inputSize = net.Layers(1).InputSize;

% Replace final layers for bird classification
lgraph = layerGraph(net);
numClasses = numel(categories(trainImgs.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'class_output')];
lgraph = replaceLayer(lgraph, 'fc1000', newLayers);

% Train the network
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Plots', 'training-progress');

net = trainNetwork(data, lgraph, options);

% Evaluate on test data
[YPred, scores] = classify(net, testImgs);
accuracy = mean(YPred == testImgs.Labels);
disp("Test Accuracy: " + accuracy);
