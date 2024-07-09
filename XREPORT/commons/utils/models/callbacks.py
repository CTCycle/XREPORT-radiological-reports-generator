import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
# Real time history callback
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):    
     
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % 2 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 5 == 0:              
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label='training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label='validation loss')
                plt.legend(loc='best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Categorical Crossentropy')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label='train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label='validation metrics') 
                plt.legend(loc='best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Categorical accuracy')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi = 300)
            plt.close() 

            
# [CALLBACK TO GENERATE REPORTS]
#==============================================================================
# Real time history callback
#==============================================================================
class GenerateTextCallback(tf.keras.callbacks.Callback):
    def __init__(self, image, sequence, tokenizer, ):       
        
        self.image = image
        self.sequence = sequence
        self.tokenizer = tokenizer
        self.start_seq = '[CLS]'        

    def on_epoch_end(self, epoch, logs=None): 
        if epoch % 1 == 0:         
            caption = self._generate_caption(self.input_image)
            print(f'\nSample caption at epoch {epoch}: {caption}')

    def _generate_caption(self, image):
        # Convert start sequence to tokens and initialize the sequence
        sequence = [self.tokenizer.convert_tokens_to_ids([self.start_seq])]
        for _ in range(self.max_len):
            # Predict the next word
            token_list = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=self.max_len, padding='post')
            preds = self.model.predict([image, token_list], verbose=0)
            next_word_token = np.argmax(preds, axis=-1)[0]
            # End loop if EOS token is predicted
            if next_word_token == self.tokenizer.word_index['[end]']:
                break
            # Append predicted word token to the sequence
            sequence[0].append(next_word_token)

        # Decode the sequence to text
        # Cleanup and format the report
        cleaned_caption = [token.replace("##", "") if token.startswith("##") else f" {token}" for token in sequence if token not in ['[CLS]', '[SEP]']]
        caption = ''.join(cleaned_caption)              
     
        return caption