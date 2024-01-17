import os
import sys
import numpy as np
import pandas as pd
import json

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and components
#------------------------------------------------------------------------------
from modules.components.data_assets import PreProcessing
from modules.components.training_assets import ModelTraining, InferenceTools
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# ....
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
XREPORT generation
-------------------------------------------------------------------------------
...
''')

# check report folder and generate list of images paths
#------------------------------------------------------------------------------
if len(os.listdir(GlobVar.predict_path)) == 0:
    print('''No XRAY scans found in the report generation folder, please add them before continuing,
the script will now be closed!''')
    sys.exit()
else:
    scan_paths = [] 
    for root, dirs, files in os.walk(GlobVar.predict_path):
        for file in files:
            scan_paths.append(os.path.join(root, file))
    print(f'''XRAY images found: {len(scan_paths)}
Report generation will start once you've selected the model.''')    

# Load pretrained model and its parameters
#------------------------------------------------------------------------------
inference = InferenceTools() 
model = inference.load_pretrained_model(GlobVar.model_path)
load_path = inference.model_path
parameters = inference.model_configuration
model.summary(expand_nested=True)

# Load the tokenizer
#------------------------------------------------------------------------------
PP = PreProcessing()
tokenizer_path = os.path.join(GlobVar.data_path, 'Tokenizers')
tokenizer = PP.load_tokenizer(tokenizer_path, 'word_tokenizer.json')

# Load images in memory
#------------------------------------------------------------------------------




# [GENERATE REPORTS]
#==============================================================================
# ....
#==============================================================================
print('''Generate the reports for XRAY images
''')

# generate input windows for predictions
#------------------------------------------------------------------------------ 
predictions_inputs = [X_times, X_extractions, X_specials]
last_timepoints = times.tail(parameters['Window size'])
last_extractions = extractions.tail(parameters['Window size'])
last_special = specials.tail(parameters['Window size'])
last_windows = [np.reshape(last_timepoints, (1, parameters['Window size'], 1)),
                np.reshape(last_extractions, (1, parameters['Window size'], 10)),   
                np.reshape(last_special, (1, parameters['Window size'], 1))]

# perform prediction with selected model
#------------------------------------------------------------------------------
probability_vectors = model.predict(predictions_inputs)
next_prob_vectors = model.predict(last_windows)

# create list of expected extractions and their probility vector
#------------------------------------------------------------------------------
expected_extractions = []
next_expected_extractions = []
for vector in probability_vectors[0]:
    super_vector = [i+1 for i, x in enumerate(vector) if x in sorted(vector, reverse=True)[:10]]    
    expected_extractions.append(super_vector)                      
expected_special = np.argmax(probability_vectors[1], axis=-1).reshape(-1, 1) + 1

for vector in next_prob_vectors[0]:
    super_vector = [i+1 for i, x in enumerate(vector) if x in sorted(vector, reverse=True)[:10]]    
    next_expected_extractions.append(super_vector)                      
next_expected_special = np.argmax(next_prob_vectors[1], axis=-1).reshape(-1, 1) + 1

# synchronize the window of timesteps with the predictions
#------------------------------------------------------------------------------ 
sync_ext_vectors = {f'Probability of {i+1} in extraction' : [] for i in range(20)}
sync_ext_values = {f'Predicted N.{i+1}' : [] for i in range(10)}
sync_sp_vectors = {f'Probability of {i+1} as special' : [] for i in range(20)}
sync_sp_values = []

for ts in range(parameters['Window size']):
    for N in sync_ext_vectors.keys():
        sync_ext_vectors[N].append('')
    for N in sync_ext_values.keys(): 
        sync_ext_values[N].append('')
    for N in sync_sp_vectors.keys(): 
        sync_sp_vectors[N].append('')
    sync_sp_values.append('') 

for x, y, s, z in zip(probability_vectors[0], probability_vectors[1], expected_extractions, expected_special):
    for i, N in enumerate(sync_ext_vectors.keys()):
        sync_ext_vectors[N].append(x[i])
    for i, N in enumerate(sync_ext_values.keys()): 
        sync_ext_values[N].append(s[i]) 
    for i, N in enumerate(sync_sp_vectors.keys()): 
        sync_sp_vectors[N].append(y[i])    
    sync_sp_values.append(z) 

for x, y, s, z in zip(next_prob_vectors[0], next_prob_vectors[1], next_expected_extractions, next_expected_special):
    for i, N in enumerate(sync_ext_vectors.keys()):
        sync_ext_vectors[N].append(x[i])
    for i, N in enumerate(sync_ext_values.keys()): 
        sync_ext_values[N].append(s[i]) 
    for i, N in enumerate(sync_sp_vectors.keys()): 
        sync_sp_vectors[N].append(y[i])    
    sync_sp_values.append(z) 

# add column with prediction to a new dataset and merge it with the input predictions
#------------------------------------------------------------------------------
df_forecast = pd.DataFrame()
for key, item in sync_ext_values.items():
    df_forecast[key] = item
df_forecast['predicted special'] = sync_sp_values    
for key, item in sync_ext_vectors.items():
    df_forecast[key] = item
for key, item in sync_sp_vectors.items():
    df_forecast[key] = item

df_merge = pd.concat([df_predictions, df_forecast], axis=1)

# print console report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------
Next predicted number series: {expected_extractions[-1]}
Next predicted special number: {expected_special[-1]}
-------------------------------------------------------------------------------
''')
print('Probability vector from softmax (%):')
for i, (x, y) in enumerate(sync_ext_vectors.items()):    
    print(f'{x} = {round((y[-1] * 100), 3)}')

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''
-------------------------------------------------------------------------------
Saving WFL prediction file
-------------------------------------------------------------------------------
''')

file_loc = os.path.join(GlobVar.fc_path, 'WFL_predictions.csv')         
df_merge.to_csv(file_loc, index=False, sep = ';', encoding = 'utf-8')





