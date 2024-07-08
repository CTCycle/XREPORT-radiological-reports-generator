import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers

from XREPORT.commons.utils.models.scheduler import LRScheduler
from XREPORT.commons.utils.models.transformers import TransformerEncoder, TransformerDecoder, SoftMaxClassifier
from XREPORT.commons.utils.models.image_encoding import ImageEncoder
from XREPORT.commons.utils.models.embeddings import PositionalEmbedding
from XREPORT.commons.utils.models.training import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.constants import CONFIG


# [XREP CAPTIONING MODEL]
#------------------------------------------------------------------------------
class XREPORTModel: 

    def __init__(self, vocab_size): 
        self.vocab_size = vocab_size
        self.img_shape = CONFIG["model"]["IMG_SHAPE"] 
        self.sequence_length = CONFIG["dataset"]["MAX_CAPTION_SIZE"] + 1       
        self.num_layers = CONFIG["model"]["NUM_LAYERS"]        
        self.learning_rate = CONFIG["training"]["LR_SCHEDULER"]["POST_WARMUP_LR"]
        self.warmup_steps = CONFIG["training"]["LR_SCHEDULER"]["WARMUP_STEPS"]
        self.xla_state = CONFIG["training"]["XLA_STATE"]  

        # initialize the image encoder and the transformers encoders and decoders
        self.image_encoder = ImageEncoder()
        self.encoders = [TransformerEncoder() for _ in range(self.num_layers)]
        self.decoders = [TransformerDecoder(self.vocab_size) for _ in range(self.num_layers)] 
        self.img_input = layers.Input(shape=self.img_shape, name='image_input')
        self.seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input') 
        self.embeddings = PositionalEmbedding(self.vocab_size) 
        self.classifier = SoftMaxClassifier(1024, self.vocab_size)     
                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True):                
       
        # encode images using the convolutional encoder
        image_features = self.image_encoder(self.img_input)      
        embeddings = self.embeddings(self.seq_input)

        # handle the connections between transformers blocks        
        encoder_output = image_features
        decoder_output = embeddings
        for encoder, decoder in zip(self.encoders, self.decoders):
            encoder_output = encoder(encoder_output, training=False)
            decoder_output = decoder(decoder_output, encoder_output, training=False)

        # apply the softmax classifier layer
        output = self.classifier(decoder_output)    

        # define the model from inputs and outputs
        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)       
        # define model compilation parameters such as learning rate, loss, metrics and optimizer
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = MaskedAccuracy()
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=self.xla_state)         
        if summary:
            model.summary(expand_nested=True)

        return model
       



