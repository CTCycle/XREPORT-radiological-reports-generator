from torch import compile as torch_compile
from keras import layers, Model, optimizers

from XREPORT.commons.utils.learning.scheduler import WarmUpLRScheduler
from XREPORT.commons.utils.learning.transformers import TransformerEncoder, TransformerDecoder, SoftMaxClassifier
from XREPORT.commons.utils.learning.encoder import BeitXRayImageEncoder
from XREPORT.commons.utils.learning.embeddings import PositionalEmbedding
from XREPORT.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.constants import TOKENIZERS_PATH
from XREPORT.commons.logger import logger


# [XREP CAPTIONING MODEL]
###############################################################################
class XREPORTModel: 

    def __init__(self, vocabulary_size, configuration):         
        self.seed = configuration.get('train_seed', 42)
        self.sequence_length = configuration.get('max_report_size', 200) 
        self.img_shape = (224,224,3) 
        
        self.embedding_dims = configuration.get('embedding_dims', 256)
        self.num_heads = configuration.get('attention_heads', 3) 
        self.num_encoders = configuration.get('num_encoders', 3) 
        self.num_decoders = configuration.get('num_decoders', 3) 
        self.freeze_img_encoder = configuration.get('freeze_img_encoder', 3)
        self.jit_compile = configuration.get('jit_compile', False)
        self.jit_backend = configuration.get('jit_backend', 'inductor')
        self.has_LR_scheduler = configuration.get('use_scheduler', False)  
        self.post_warm_lr = configuration.get('post_warmup_LR', 0.0001)
        self.warmup_steps = configuration.get('warmup_steps',10000)
        self.temperature = configuration.get('training_temperature', 1.0)
        self.configuration = configuration
        self.vocabulary_size = vocabulary_size
        
        # initialize the image encoder and the transformers encoders and decoders
        self.img_input = layers.Input(shape=self.img_shape, name='image_input')
        self.seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input')         
        
        self.img_encoder = BeitXRayImageEncoder(
            self.freeze_img_encoder, self.embedding_dims)
        
        self.encoders = [TransformerEncoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.decoders = [TransformerDecoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_decoders)]        
        self.embeddings = PositionalEmbedding(
            self.vocabulary_size, self.embedding_dims, self.sequence_length) 
        self.classifier = SoftMaxClassifier(
            1024, self.vocabulary_size, self.temperature) 

    #--------------------------------------------------------------------------
    def compile_model(self, model, model_summary=True):
        lr_schedule = self.post_warm_lr
        if self.has_LR_scheduler:            
            post_warmup_LR = self.configuration.get('post_warmup_LR', 40000)   
            warmup_steps = self.configuration.get('warmup_steps', 1000)                        
            lr_schedule = WarmUpLRScheduler(post_warmup_LR, warmup_steps)                  
        
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = [MaskedAccuracy()]
        opt = optimizers.AdamW(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)                 
  
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode='default')

        return model                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
        # encode images and extract their features using the convolutional 
        # image encoder or a selected pretrained model
        image_features = self.img_encoder(self.img_input)      
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
        model = self.compile_model(model, model_summary=model_summary)   

        return model
       



