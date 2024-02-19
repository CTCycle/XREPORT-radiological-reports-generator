import sys
import art

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('XREPORT')
print(ascii_art)

# [MAIN MENU]
# =============================================================================
# Starting DITK analyzer, checking for dictionary presence and perform conditional
# import of modules
# =============================================================================
user_operations = UserOperations()
operations_menu = {'1': 'Data validation',
                   '2': 'Pretrain XREPORT model',
                   '3': 'Evaluate XREPORT model',                   
                   '4': 'Generate reports based on images',
                   '5': 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print() 
    if op_sel == 1:
        import modules.data_validation
        del sys.modules['modules.data_validation']      
    elif op_sel == 2:
        import modules.model_training
        del sys.modules['modules.XREPORT_training']
    elif op_sel == 3:
        import modules.model_evaluation
        del sys.modules['modules.model_evaluation']
    elif op_sel == 4:
        import modules.report_generator
        del sys.modules['modules.XREPORT_generator']
    elif op_sel == 5:
        break


