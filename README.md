# XREPORT: Radiological Reports Generation

## 1. Project Overview
XRAY Report Generator is a machine learning-based tool designed to assist radiologists in generating descriptive reports from X-ray images. This project aims to reduce the time and effort required by radiologists to write detailed reports based on the XRAY scan description, thereby increasing efficiency and turnover. The generative model is trained using combinations of XRAY images and their labels (descriptions), in the same fashion as image captioning models learn a sequence of word tokens associated to specific parts of the image. While originally developed around the MIMIC-CXR Database (https://www.kaggle.com/datasets/wasifnafee/mimic-cxr), this project can be applied to any dataset with X-ray scans labeled with their respective radiological reports (or any kind of description). The XREPORT Deep Learning (DL) model developed for this scope makes use of a transformer encoder-decoder architecture, which relies on both self attention and cross attention to improve text significance within the clinical image context. The images features are extracted using a custom convolutional encoder with pooling layers to reduce dimensionality. Once a pretrained model is obtained leveraging a large number of X-RAY scans and their descriptions, the model can be used in inference mode to generate radiological reports from the raw pictures. 

## 2. XREPORT model
The XREPORT model is based on a transformer encoder-decoder architecture. Three stacked encoders with multi-head self-attention and feedforward networks are used downstream to the convolutional image encoder network to generate vectors with extracted x-ray scan features. The X-RAY scans are processed and reduced in dimensionality using a series of convolutional layers followed by max-pooling operations. These image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs, to determine most important features in the images associated with specific words in the text. To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.T

**DistilBERT tokenization:** to improve the vectorization and the semantic representation of the training text corpus, the pretrained tokenizer of the DistilBERT model has been used to split text into subwords and vectorize the tokens. The base model is taken from `distilbert/distilbert-base-uncased`, and is automatically downloaded in `training/BERT`. Once saved, the weights are loaded each time a new training session is called. The XREPORT model performs word embedding by coupling token embeddings with positional embeddings, and supports masking for variable-length sequences, ensuring adaptability to text sequences of different length.

**XREP transformers:** the body of the model comprises a series of transformer encoders/decoders. The transformer encoder employs multi-head self-attention and feedforward networks to further process the encoded images. These transformed image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs. To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.

## 3. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is properly installed on your system before proceeding.

- To set up the environment, run `scripts/environment_setup.bat`. This script installs Keras 3 with pytorch support as backend, and includes includes all required CUDA dependencies to enable GPU utilization (CUDA 12.1).
- **IMPORTANT:** run `scripts/package_setup.bat` if the path to the project folder is changed for any reason after installation, or the app won't work!

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. 

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 4. How to use
The project is organized into subfolders, each dedicated to specific tasks.

### Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

- **dataset:** contains images used to train the XREPORT model (`dataset/images`), as well as the file `XREP_dataset.csv` that should be provided for training purposes. Within this .csv file, the images name should be located under the "id" column and the corresponding text should be in the "text" column. 

- **generation:** 
- `input_images:` this is where you place images intended for inference using the pretrained XREPORT model.
- `reports:` the generated radiological reports from input images are saved within this folder. 

- **results:** used to save the results of data validation processes. This helps in keeping track of validation metrics and logs.

- **checkpoints:** pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

### Inference
Here you can find the necessary files to run pretrained models in inference mode, and use them to generate radiological reports from input X-ray scans.

- Run `images_encoding.py` to use the pretrained encoder from a model checkpoint to extract abstract representation of image features in the form of lower-dimension embeddings. 

### Training
This folder contains the necessary files for conducting model training and evaluation. 
- Run `model_training.py` to initiate the training process for the autoencoder

### Validation
Data validation and pretrained model evaluations are performed using the scripts within this folder.
- Launch the jupyter notebook `model_evaluation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics.
- Launch the jupyter notebook `data_validation.ipynb` to validate the available data with different metrics.

### 4.1 Configurations
For customization, you can modify the main configuration parameters using `configurations.json` in the root project folder. 

#### Dataset Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SAMPLE_SIZE        | Number of samples to use from the dataset                |
| VALIDATION_SIZE    | Proportion of the dataset to use for validation          |
| IMG_NORMALIZE      | Whether to normalize image data                          |
| IMG_AUGMENT        | Whether to apply data augmentation to images             |
| MAX_REPORT_SIZE    | Max length of text report                                |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |

#### Model Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| IMG_SHAPE          | Shape of the input images (height, width, channels)      |
| EMBEDDING_DIMS     | Embedding dimensions (valid for both models)             |  
| NUM_HEADS          | Number of attention heads                                | 
| NUM_ENCODERS       | Number of encoder layers                                 |
| NUM_DECODERS       | Number of decoder layers                                 |
| SAVE_MODEL_PLOT    | Whether to save a plot of the model architecture         |

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| LEARNING_RATE      | Learning rate for the optimizer                          |
| BATCH_SIZE         | Number of samples per batch                              |
| MIXED_PRECISION    | Whether to use mixed precision training                  |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| XLA_STATE          | Whether to enable XLA (Accelerated Linear Algebra)       |
| ML_DEVICE          | Device to use for training (e.g., GPU)                   |
| NUM_PROCESSORS     | Number of processors to use for data loading             |         

#### Evaluation Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| BATCH_SIZE         | Number of samples per batch during evaluation            | 
| SAMPLE_SIZE        | Number of samples from the dataset (evaluation only)     |
| VALIDATION_SIZE    | Fraction of validation data (evaluation only)            |


## 5. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

