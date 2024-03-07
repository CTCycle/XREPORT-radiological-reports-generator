# XRAYREP: Radiological Reports Generation

## Project Overview
XRAY Report Generator is a machine learning-based tool designed to assist radiologists in generating descriptive reports from X-ray images. This project aims to reduce the time and effort required by radiologists to write detailed reports based on the XRAY scan description, thereby increasing efficiency and turnover. The generative model is trained using combinations of XRAY images and their labels (descriptions), in the same fashion as image captioning models learn a sequence of word tokens associated to specific parts of the image. The XREPORT Deep Learning (DL) model developed for this scope makes use of a transformer encoder-decoder architecture, which relies on both self attention and cross attention to improve text significance within the clinical image context. The images features are extracted using a custom convolutional encoder with pooling layers to reduce dimensionality. Once a pretrained model is obtained leveraging a large number of X-RAY scans and their descriptions, the model can be used in inference mode to generate radiological reports from the raw pictures. 

## XREP Captioning model
The XREP Captioning Model is based on an advanced deep learning architecture tailored for generating radiological reports from X-RAY scans. Leveraging state-of-the-art captioning techniques, this model combines the power of multi-layer convolutional neural networks (CNNs) for image feature extraction with a transformer-based architecture for sequence generation. The X-RAY scans are processed and reduced in dimensionality using a series of convolutional layers followed by max-pooling operations. This process encodes each X-RAY image into a compact feature vector, capturing essential information that are associated the the words in the radiological reports. 

**BioBERT embeddings:** to improve the vectorization and the semantic representation of the training text corpus, the pretrained embeddings of the BioBERT model have been integrated into the model architecture, and will be finetuned while training the XREP-captioning model. The base model is taken from `dmis-lab/biobert-base-cased-v1.1` (https://huggingface.co/dmis-lab/biobert-base-cased-v1.1), and is automatically downloaded in `training/BioBERT` the first time this script is run. Once saved, the weights are loaded each time a new training session is called. In addition to this custom embedding pipeline, the model implements positional embedding by seamlessly blending token embeddings with positional embedding vectors based on the word indexing in the reports. It also supports masking for variable-length sequences, ensuring adaptability to text sequences of different length.

**XREP transformers:** the body of the model comprises a series of transformer encoders/decoders. The transformer encoder employs multi-head self-attention and feedforward networks to further process the encoded images. These transformed image vectors are then fed into the transformer decoder, which applies cross-attention between encoder and decoder inputs. To ensure coherent report generation, the model employs causal masking on token sequences during decoding. This auto-regressive mechanism guarantees that generated reports consider the context of previously generated tokens.

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


## How to use
The project is organized into subfolders, each dedicated to specific tasks. The `utils/` folder houses crucial components utilized by various scripts. It's critical to avoid modifying these files, as doing so could compromise the overall integrity and functionality of the program.

**Data:** this folder contains the data used for the model training, which should include a folder with X-ray images and a .csv file reporting the images name and related radiological reports. X-ray scan must be loaded in `data/images`.
Run the jupyter notebook `data_validation.ipynb` to perform in-depth analysis of the image-to-report dataset, with the results being saved in `data/validation`. 

**Training:** contains the necessary files for conducting model training and evaluation. `model/checkpoints` acts as the default repository where checkpoints of pre-trained models are stored. Run `model_training.py` to initiate the training process for deep learning models, or launch `model_evaluation.py` to evaluate the performance metrics of pre-trained models.

**Inference:** use `report_generator.py` to load pretrain model checkpoints and run them in inference mode. Generate radiological reports from the source X-ray images located within `inference/reports`. The reports are saved as .csv file in the same directory.

### Configurations
The configurations.py file allows to change the script configuration. The following parameters are available:

**Advanced settings for training:**
- `use_mixed_precision:` whether or not to use mixed precision for faster training (mix float16/float32)
- `use_tensorboard:` activate or deactivate tensorboard logging
- `XLA_acceleration:` use of linear algebra acceleration for faster training 
- `training_device:` select the training device (CPU or GPU)
- `num_processors:` number of processors (cores) to be used during training; if set to 1, multiprocessing is not used

**Settings for training routine:**
- `epochs:` number of training iterations
- `learning_rate:` learning rate of the model 
- `batch_size:` size of batches to be fed to the model during training

**Model settings:**
- `picture_shape:` full shape of the images as (height, width, channels)
- `embedding_size:` embedding dimensions (valid for both models)
- `kernel_size:` size of convolutional kernel (image encoder)
- `num_heads:` number of attention heads
- `generate_model_graph:` generate and save 2D model graph (as .png file)

**Settings for training data:**
- `num_train_samples:` number of images to use for the model training 
- `num_test_samples:` number of samples to use as validation data
- `augmentation:` whether or not to perform data agumentation on images (significant impact on training time)

**General settings:**
- `seed:` global random seed
- `split_seed:` seed of dataset splitting

**Number of samples and batch size:** This application is designed for efficient on-the-fly image loading using a custom generator. Therefore, it is advisable to carefully choose the number of training and testing samples and adjust the batch size accordingly.

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

