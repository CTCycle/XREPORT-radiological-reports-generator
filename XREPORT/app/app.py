import sys
from PySide6.QtWidgets import QApplication

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from XREPORT.app.interface.window import MainWindow
from XREPORT.app.constants import UI_PATH, QSS_PATH


# [RUN MAIN]
###############################################################################
if __name__ == "__main__":  
    app = QApplication(sys.argv) 

    # ---- LOAD QSS STYLE ----
    with open(QSS_PATH, "r") as f:
        app.setStyleSheet(f.read())

    main_window = MainWindow(UI_PATH)   
    main_window.show()
    sys.exit(app.exec())


   
