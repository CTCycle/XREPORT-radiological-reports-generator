import sys
from PySide6.QtWidgets import QApplication

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from TokenBenchy.commons.interface.window import MainWindow
from TokenBenchy.commons.constants import UI_PATH
from TokenBenchy.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == "__main__":  
    app = QApplication(sys.argv) 
    main_window = MainWindow(UI_PATH)   
    main_window.show()
    sys.exit(app.exec())

   
