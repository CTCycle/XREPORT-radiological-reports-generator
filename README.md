# XREPORT: Radiological Reports Generation

## 1. Project Overview
XRAY Report Generator is a machine learning-based tool designed to assist radiologists in generating descriptive reports from X-ray images. This project aims to reduce the time and effort required by radiologists to write detailed reports based on the XRAY scan description, thereby increasing efficiency and turnover. The generative model is trained using combinations of XRAY images and their labels (descriptions), in the same fashion as image captioning models learn a sequence of word tokens associated to specific parts of the image. The XREPORT Deep Learning (DL) model developed for this scope makes use of a transformer encoder-decoder architecture, which relies on both self attention and cross attention to improve text significance within the clinical image context. The images features are extracted using a custom convolutional encoder with pooling layers to reduce dimensionality. Once a pretrained model is obtained leveraging a large number of X-RAY scans and their descriptions, the model can be used in inference mode to generate radiological reports from the raw pictures. 

## 2. XREPORT model
The XREPORT model is based on a transformer encoder-decoder architecture. Three stacked encoders with multi-head self-attention and feedforward networks are used downstream to the convolutional image encoder network to generate vectors with extracted x-ray scan features. The X-RAY scans are processed and reduced in dimensionality using a series of convolutional layers followed by max-pooling operations. These image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs, to determine most important features in the images associated with specific words in the text. To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.T

**DistilBERT tokenization:** to improve the vectorization and the semantic representation of the training text corpus, the pretrained tokenizer of the DistilBERT model has been used to split text into subwords and vectorize the tokens. The base model is taken from `distilbert/distilbert-base-uncased`, and is automatically downloaded in `training/BERT`. Once saved, the weights are loaded each time a new training session is called. The XREPORT model performs word embedding by coupling token embeddings with positional embeddings, and supports masking for variable-length sequences, ensuring adaptability to text sequences of different length.

**XREP transformers:** the body of the model comprises a series of transformer encoders/decoders. The transformer encoder employs multi-head self-attention and feedforward networks to further process the encoded images. These transformed image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs. To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.

## 3. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is installed on your system before proceeding.

- To set up a CPU-only environment, run `setup/create_cpu_environment.bat`. This script installs the base version of TensorFlow, which is lighter and does not include CUDA libraries.
- For GPU support, which is necessary for model training on a GPU, use `setup/create_gpu_environment.bat`. This script includes all required CUDA dependencies to enable GPU utilization.
- Once the environment has been created, run `scripts/package_setup.bat` to install the app package locally.
- **IMPORTANT:** run `scripts/package_setup.bat` if you move the project folder somewhere else after installation, or the app won't work! 

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. 

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 4. How to use
The project is organized into subfolders, each dedicated to specific tasks. The `XREPORT/utils` folder houses crucial components utilized by various scripts. It's critical to avoid modifying these files, as doing so could compromise the overall integrity and functionality of the program.

**Data:** this folder contains the data used for the model training, which should include a folder with X-ray images and a .csv file reporting the images name and related radiological reports. X-ray scan must be loaded in `XREPORT/data/images`.
Run the jupyter notebook `XREPORT/data_validation.ipynb` to perform Explorative Data analysis (EDA) of the dataset, with the results being saved in `XREPORT/data/validation`. 

**Training:** contains the necessary files for conducting model training and evaluation. `XREPORT/model/checkpoints` acts as the default repository where checkpoints of pre-trained models are stored. Run `model_training.py` to initiate the training process for deep learning models, or launch `model_evaluation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics.

**Inference:** use `report_generator.py` to load pretrain model checkpoints and run them in inference mode. Generate radiological reports from the source X-ray images located within `XREPORT/inference/reports`. The reports are saved as .csv file in the same directory.

### 4.1 Configurations
The configurations.py file allows to change the script configuration. 

| Category                | Setting                | Description                                                       |
|-------------------------|------------------------|-------------------------------------------------------------------|
| Advanced settings       | USE_MIXED_PRECISION    | use mixed precision for faster training (float16/32)              |
|                         | USE_TENSORBOARD        | Activate/deactivate tensorboard logging                           |
|                         | XLA_STATE              | Use linear algebra acceleration for faster training               |
|                         | ML_DEVICE              | Select the training device (CPU or GPU)                           |
|                         | NUM_PROCESSORS         | Number of processors (cores) to use; 1 disables multiprocessing   |
| Training routine        | EPOCHS                 | Number of training iterations                                     |
|                         | LEARNING_RATE          | Learning rate of the model                                        |
|                         | BATCH_SIZE             | Size of batches for model training                                |
| Model settings          | IMG_SHAPE              | Full shape of the images as (height, width, channels)             |
|                         | EMBEDDING_DIMS         | Embedding dimensions (valid for both models)                      |
|                         | KERNEL_SIZE            | Size of convolutional kernel (image encoder)                      |
|                         | NUM_HEADS              | Number of attention heads                                         |
|                         | SAVE_MODEL_PLOT        | Generate/save 2D model graph (as .png file)                       |
| Training data           | TRAIN_SAMPLES          | Number of images for model training                               |
|                         | TEST_SAMPLES           | Number of samples for validation data                             |
|                         | IMG_AUGMENT            | Perform data augmentation on images (affects training time)       |
| General settings        | SEED                   | Global random seed                                                |
|                         | SPLIT_SEED             | Seed for dataset splitting                                        |

## 5. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

