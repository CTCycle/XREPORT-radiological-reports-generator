import keras

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger



###############################################################################
def evaluation_report(model : keras.Model, train_dataset, validation_dataset):
    
    train_eval = model.evaluate(train_dataset, verbose=1)
    validation_eval = model.evaluate(validation_dataset, verbose=1)
    print('Train dataset:')
    print(f'Loss: {train_eval[0]}')    
    print(f'Metric: {train_eval[1]}')  
    print('Test dataset:')
    print(f'Loss: {validation_eval[0]}')    
    print(f'Metric: {validation_eval[1]}')



        

              
        
        
            
        
