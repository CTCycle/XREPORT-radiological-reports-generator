import sys
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.app.interface.window import MainWindow
from XREPORT.app.constants import UI_PATH


# [RUN MAIN]
###############################################################################
if __name__ == "__main__":  
    app = QApplication(sys.argv) 

    # setup stylesheet
    extra = {'density_scale': '-1'}
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra)

    main_window = MainWindow(UI_PATH)   
    main_window.show()
    sys.exit(app.exec())


   
