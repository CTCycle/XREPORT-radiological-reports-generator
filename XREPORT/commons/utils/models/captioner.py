import keras
from keras import layers, Model

from XREPORT.commons.utils.models.scheduler import LRScheduler
from XREPORT.commons.utils.models.transformers import TransformerEncoder, TransformerDecoder, SoftMaxClassifier
from XREPORT.commons.utils.models.image_encoding import ImageEncoder
from XREPORT.commons.utils.models.embeddings import PositionalEmbedding
from XREPORT.commons.utils.models.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger


# [XREP CAPTIONING MODEL]
###############################################################################
class XREPORTModel: 

    def __init__(self, vocab_size): 
        self.vocab_size = vocab_size
        self.img_shape = CONFIG["model"]["IMG_SHAPE"] 
        self.sequence_length = CONFIG["dataset"]["MAX_REPORT_SIZE"] - 1 
        self.embedding_dims = CONFIG["model"]["EMBEDDING_DIMS"] 
        self.num_heads = CONFIG["model"]["NUM_HEADS"]  
        self.num_encoders = CONFIG["model"]["NUM_ENCODERS"]   
        self.num_decoders = CONFIG["model"]["NUM_ENCODERS"]      
        self.learning_rate = CONFIG["training"]["LR_SCHEDULER"]["POST_WARMUP_LR"]
        self.warmup_steps = CONFIG["training"]["LR_SCHEDULER"]["WARMUP_STEPS"]
        self.xla_state = CONFIG["training"]["XLA_STATE"]  

        # initialize the image encoder and the transformers encoders and decoders
        self.img_input = layers.Input(shape=self.img_shape, name='image_input')
        self.seq_input = layers.Input(shape=(self.sequence_length,), name='seq_input') 
        self.image_encoder = ImageEncoder()
        self.encoders = [TransformerEncoder(self.embedding_dims, self.num_heads) for _ in range(self.num_encoders)]
        self.decoders = [TransformerDecoder(self.embedding_dims, self.num_heads) for _ in range(self.num_decoders)]        
        self.embeddings = PositionalEmbedding(self.vocab_size, self.embedding_dims, self.sequence_length) 
        self.classifier = SoftMaxClassifier(1024, self.vocab_size)     
                

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True):                
       
        # encode images using the convolutional encoder
        image_features = self.image_encoder(self.img_input)      
        embeddings = self.embeddings(self.seq_input) 
        padding_mask = self.embeddings.compute_mask(self.seq_input) 
        
        # handle the connections between transformers blocks        
        encoder_output = image_features
        decoder_output = embeddings
    
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output, 
                                     training=False, mask=padding_mask)

        # apply the softmax classifier layer
        output = self.classifier(decoder_output)    

        # define the model from inputs and outputs
        model = Model(inputs=[self.img_input, self.seq_input], outputs=output)       
        # define model compilation parameters such as learning rate, loss, metrics and optimizer
        lr_schedule = LRScheduler(self.learning_rate, warmup_steps=10)
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = [MaskedAccuracy()]
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=self.xla_state)         
        if summary:
            model.summary(expand_nested=True)

        return model
       



