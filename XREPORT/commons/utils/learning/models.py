import torch
from keras import layers, Model, optimizers
from transformers import AutoFeatureExtractor, AutoModel

from XREPORT.commons.utils.learning.scheduler import LRScheduler
from XREPORT.commons.utils.learning.transformers import TransformerEncoder, TransformerDecoder, SoftMaxClassifier
from XREPORT.commons.utils.learning.encoder import ImageEncoder
from XREPORT.commons.utils.learning.embeddings import PositionalEmbedding
from XREPORT.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.constants import TOKENIZERS_PATH, CONFIG
from XREPORT.commons.logger import logger


# [XREP CAPTIONING MODEL]
###############################################################################
class XREPORTModel: 

    def __init__(self, vocabulary_size, configuration): 
        self.vocabulary_size = vocabulary_size
        self.seed = configuration["SEED"]
        self.sequence_length = configuration["dataset"]["MAX_REPORT_SIZE"] - 1 
        self.img_shape = configuration["model"]["IMG_SHAPE"] 
        self.pretrained_encoder = configuration["model"]["PRETRAINED_IMG_ENCODER"]
        self.embedding_dims = configuration["model"]["EMBEDDING_DIMS"] 
        self.num_heads = configuration["model"]["NUM_HEADS"]  
        self.num_encoders = configuration["model"]["NUM_ENCODERS"]   
        self.num_decoders = configuration["model"]["NUM_DECODERS"]
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]             
        self.learning_rate = configuration["training"]["LR_SCHEDULER"]["POST_WARMUP_LR"]
        self.warmup_steps = configuration["training"]["LR_SCHEDULER"]["WARMUP_STEPS"]
        self.configuration = configuration
        
        # initialize the image encoder and the transformers encoders and decoders
        self.img_input = layers.Input(shape=self.img_shape, name='image_input')
        self.seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input')         
        
        self.image_encoder, self.feature_extractor = self.initialize_image_encoder(self.pretrained_encoder)

        self.encoders = [TransformerEncoder(self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.decoders = [TransformerDecoder(self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_decoders)]        
        self.embeddings = PositionalEmbedding(self.vocabulary_size, self.embedding_dims, self.sequence_length) 
        self.classifier = SoftMaxClassifier(1024, self.vocabulary_size)

    # build model given the architecture
    #--------------------------------------------------------------------------
    def initialize_image_encoder(self, encoder_name):

        feature_extractor = None
        encoder_name = 'custom' if encoder_name is None else encoder_name        
        encoder_mapping = {"custom": None,
                           "vit-base": "google/vit-base-patch16-224",
                           "convnext-base": "facebook/convnext-base-224",
                           "swin-tiny": "microsoft/swin-tiny-patch4-window7-224"}
        
        selected_model = encoder_mapping[encoder_name]
        if encoder_name not in encoder_mapping or selected_model is None:
            logger.warning(f"Unsupported or unspecified image encoder: {encoder_name}. Falling back to custom image encoder.")
            encoder = ImageEncoder()            
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model, cache_dir=TOKENIZERS_PATH)
            encoder = AutoModel.from_pretrained(selected_model, cache_dir=TOKENIZERS_PATH)

        return encoder, feature_extractor                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):                
       
        # encode images and extract their features using the convolutional 
        # image encoder or a selected pretrained model
        image_features = self.image_encoder(self.img_input)      
        embeddings = self.embeddings(self.seq_input) 
        padding_mask = self.embeddings.compute_mask(self.seq_input)         
                
        encoder_output = image_features
        decoder_output = embeddings    
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output, 
                                     training=False, mask=padding_mask)

        # apply the softmax classifier layer
        output = self.classifier(decoder_output)    

        # wrap the model and compile it with AdamW optimizer
        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)       
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = [MaskedAccuracy()]
        opt = optimizers.AdamW(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False) 

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')       

        if model_summary:
            model.summary(expand_nested=True)

        return model
       



