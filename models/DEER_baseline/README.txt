DEER Baseline

Purpose:
    - Get a model running to serve as a starting point for the project
    - Train on the full dataset on purpose in order to ensure the model has the capacity to learn our data
    - Put off worries about generalization for later versions

Dataset:
    - full MEM dataset, no split
    - target_scanpath_length = 4 (only consider first 4 fixations per scanpath)

Modules:
    Image Encoder:
        - DenseNet201 backbone
        - Identity head
        - 1920 -> 256 Projection Layer with ReLU

    Scanpath Encoder:
        - GRU:
            - Input Size: 4
            - Hidden Size: 128
        - 128 -> 256 Projection Layer with ReLU

    The projection layers were intended to balance the contribution of both modules to the overall prediction

    Fusion:
        - Concatenation of image and scanpath features along dimension 1 (vertical)
        - 256 + 256 = 512 
    
    Task Classifier (MLP):
        - 512 -> 256 FC Layer with ReLU
        - 256 -> num-categories FC layer (output layer)
    
Preprocessing:
    Image: 
        - resize to 224x224
        - normalize using ImageNet mean and standard deviation

    Scanpath features:
        x location:
            - normalized by original image width      
        y location:
            - normalized by original image height
        fixation duration:
            - zscore normalized
        saccade amplitude:
            - zscore normalized

Training:
    Criterion: Cross Entropy Loss
    Optimizer: Adam
    Epochs: 100
    Strategy:
        - freeze Image Encoder
        - train Scanpath Encoder and Task Classifier on full dataset

Notes:
    The learning curves dislay memorization, which is great news for the purpose of this experiement
    This shows that the model has the capacity to fit the dataset, and that the training pipeline is working correctly

Thoughts for next iteration:
    - Splitting by subject / images
    - Feature engineering that follows existing literature (or even testing different features)