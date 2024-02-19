import os

# define absolute path of folders
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
rep_path = os.path.join(data_path, 'reports')
val_path = os.path.join(data_path, 'validation')   
images_path = os.path.join(data_path, 'images') 

# create folders if they do not exist
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(models_path):
    os.mkdir(models_path)
if not os.path.exists(rep_path):
    os.mkdir(rep_path)
if not os.path.exists(val_path):
    os.mkdir(val_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)

# placeholder for model save folder
#------------------------------------------------------------------------------
model_folder_path = ''
model_folder_name = ''
