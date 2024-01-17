import sys

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# [MAIN MENU]
# =============================================================================
# Starting DITK analyzer, checking for dictionary presence and perform conditional
# import of modules
# =============================================================================
print('''
-------------------------------------------------------------------------------
XRAY REPORTER
-------------------------------------------------------------------------------
...
''')
user_operations = UserOperations()
operations_menu = {'1': 'Preprocess XRAY dataset', 
                   '2': 'Pretrain XREP model',                   
                   '3': 'Generate reports based on images',
                   '4': 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()    
    if op_sel == 1:
        import modules.XREPORT_preprocessing
        del sys.modules['modules.XREPORT_preprocessing']        
    elif op_sel == 2:
        import modules.XREPORT_training
        del sys.modules['modules.XREPORT_training']
    elif op_sel == 3:
        import modules.XREPORT_generator
        del sys.modules['modules.XREPORT_generator']
    elif op_sel == 4:
        break


