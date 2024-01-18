# XRAY-report-generator

## Project description
XRAY Report Generator is a machine learning-based tool designed to assist radiologists in generating descriptive reports from X-ray images. This project aims to reduce the time and effort required by radiologists to write detailed reports based on the XRAY scan description, thereby increasing efficiency and turnover. The generative model is trained using combinations of XRAY images and their labels (descriptions), in the same fashion as image captioning is performed. The Deep Learning (DL) model developed for this scope makes use of the attention mechanisms to improve text significance within the clinical image context. As such, the core feature of this project is generating descriptive captions for X-ray images using a state-of-the-art model. The script uses a machine learning model trained on a large dataset of X-ray images and corresponding radiology reports. When an X-ray image is input into the system, the model generates a caption that describes the key findings in the image. This caption can then be used as a starting point for writing the radiology report.

## XREP model
The XREP Captioning Model is an advanced deep learning architecture tailored for generating radiological reports from X-RAY scans. Leveraging state-of-the-art captioning techniques, this model combines the power of multi-layer convolutional neural networks (CNNs) for image feature extraction with a transformer-based architecture for sequence generation.

### Key Features

**Image Feature Extraction**
X-RAY scans are processed and dimensionality-reduced using a series of max-pooling layers. This process encodes each X-RAY scan into a compact feature vector, capturing essential information for generating detailed reports.

**Positional Embedding**
The model implements positional embedding by seamlessly blending token embeddings with positional embeddings. It also supports masking for variable-length sequences, ensuring adaptability to diverse radiological reports.

**Transformer-Based Architecture**
The heart of the model comprises a transformer encoder and decoder. The transformer encoder employs multi-head self-attention and feedforward blocks to further process the encoded images. These transformed image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs.

**Auto-Regressive Sequence Generation**
To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.

## How to use
Run the XRAYREP.py file to launch the script and use the main menu to navigate the different options. From the main menu, you can select one of the following options:

**1) Pretrain XREP model** Preprocess data and pretrain the XREP captioning model 

**2) Generate reports based on images** Use a pretrained model to generate reports from raw X-RAY images

**3) Exit and close**

### Configurations
The configurations.py file allows to change the script configuration. The following parameters are available:

**Settings for training performance and monitoring options:**
- `generate_model_graph:` generate and save 2D model graph (as .png file)
- `use_mixed_precision:` whether or not to use mixed precision for faster training (mix float16/float32)
- `use_tensorboard:` activate or deactivate tensorboard logging
- `XLA_acceleration:` use of linear algebra acceleration for faster training 

**Settings for pretraining parameters:**
- `training_device:` select the training device (CPU or GPU)
- `epochs:` number of training iterations
- `learning_rate:` learning rate of the model during training
- `batch_size:` size of batches to be fed to the model during training
- `embedding_size:` embedding dimensions (valid for both models)
- `kernel_size:` size of convolutional kernel (image encoder)
- `num_heads:` number of attention heads

**Settings for data preprocessing and predictions:**
- `picture_size:` scaled size of the x-ray images
- `num_channels:` number of channels per image (set to 1 for gray-scale, 3 for RGB)
- `image_shape:` automatically calculated full image shape
- `num_samples:` number of images to consider prior to generating train and test datasets
- `test_size:` fraction of num_samples to use as validation data

### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `keras==2.10.0`
- `matplotlib==3.7.2`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `scikit-learn==1.3.0`
- `seaborn==0.12.2`
- `tensorflow==2.10.0`
- `xlrd==2.0.1`
- `XlsxWriter==3.1.3`
- `pydot==1.4.2`
- `graphviz==0.20.1`

These dependencies are specified in the provided `requirements.txt` file to ensure full compatibility with the application. 

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

## Disclaimer
...
