# XRAY-report-generator

## Project description
XRAY Report Generator is a machine learning-based tool designed to assist radiologists in generating descriptive reports from X-ray images. This project aims to reduce the time and effort required by radiologists to write detailed reports based on the XRAY scan description, thereby increasing efficiency and turnover. The generative model is trained using combinations of XRAY images and their labels (descriptions), in the same fashion as image captioning models learn a sequence of word tokens associated to specific parts of the image. The XREPORT Deep Learning (DL) model developed for this scope makes use of a transformer encoder-decoder architecture, which relies on both self attention and cross attention to improve text significance within the clinical image context. The images features are extracted using a custom convolutional encoder with pooling layers to reduce dimensionality. Once a pretrained model is obtained leveraging a large number of X-RAY scans and their descriptions, the model can be used in inference mode to generate radiological reports from the raw pictures. 

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
- `data_augmentation:` whether or not to perform data agumentation on images (significant impact on training time)

## Installation 
First, ensure that you have Python 3.10.12 installed on your system. Then, you can easily install the required Python packages using the provided requirements.txt file:

`pip install -r requirements.txt` 

In addition to the Python packages, certain extra dependencies may be required for specific functionalities. These dependencies can be installed using conda or other external installation methods, depending on your operating system. Specifically, you will need to install graphviz and pydot to enable the visualization of the 2D model architecture:
- graphviz version 2.38.0
- pydot version 1.4.2

You can install these dependencies using the appropriate package manager for your system. For instance, you might use conda or an external installation method based on your operating system's requirements.

## CUDA GPU Support (Optional, for GPU Acceleration)
If you have an NVIDIA GPU and want to harness the power of GPU acceleration using CUDA, please follow these additional steps. The application is built using TensorFlow 2.10.0 to ensure native Windows GPU support, so remember to install the appropriate versions:

### 1. Install NVIDIA CUDA Toolkit (Version 11.2)

To enable GPU acceleration, you'll need to install the NVIDIA CUDA Toolkit. Visit the [NVIDIA CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads) and select the version that matches your GPU and operating system. Follow the installation instructions provided. Alternatively, you can install `cuda-toolkit` as a package within your environment.

### 2. Install cuDNN (NVIDIA Deep Neural Network Library, Version 8.1.0.77)

Next, you'll need to install cuDNN, which is the NVIDIA Deep Neural Network Library. Visit the [cuDNN download page](https://developer.nvidia.com/cudnn) and download the cuDNN library version that corresponds to your CUDA version (in this case, version 8.1.0.77). Follow the installation instructions provided.

### 3. Additional Package (If CUDA Toolkit Is Installed)

If you've installed the NVIDIA CUDA Toolkit within your environment, you may also need to install an additional package called `cuda-nvcc` (Version 12.3.107). This package provides the CUDA compiler and tools necessary for building CUDA-enabled applications.

By following these steps, you can ensure that your environment is configured to take full advantage of GPU acceleration for enhanced performance.                 

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

