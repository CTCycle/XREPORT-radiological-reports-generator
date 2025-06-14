import sys
from PySide6.QtWidgets import QApplication

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.commons.interface.window import MainWindow
from XREPORT.commons.constants import UI_PATH
from XREPORT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == "__main__":  
    app = QApplication(sys.argv) 
    main_window = MainWindow(UI_PATH)   
    main_window.show()
    sys.exit(app.exec())


   
