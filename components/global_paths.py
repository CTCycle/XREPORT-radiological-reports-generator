import os

# define absolute path of folders
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
inference_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inference')

# create folders if they do not exist
#------------------------------------------------------------------------------
os.mkdir(data_path) if not os.path.exists(data_path) else None
os.mkdir(model_path) if not os.path.exists(model_path) else None
os.mkdir(inference_path) if not os.path.exists(inference_path) else None

