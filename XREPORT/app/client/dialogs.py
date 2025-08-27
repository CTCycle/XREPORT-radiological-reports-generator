import os
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLineEdit,
    QLabel,
    QDialogButtonBox,
    QListWidget,
)

from XREPORT.app.constants import CONFIG_PATH


###############################################################################
class SaveConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Configuration As")
        self.layout = QVBoxLayout(self)

        self.label = QLabel("Enter a name for your configuration:", self)
        self.layout.addWidget(self.label)

        self.name_edit = QLineEdit(self)
        self.layout.addWidget(self.name_edit)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_name(self):
        return self.name_edit.text().strip()


###############################################################################
class LoadConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Configuration")
        self.layout = QVBoxLayout(self)

        self.label = QLabel("Select a configuration:", self)
        self.layout.addWidget(self.label)

        self.config_list = QListWidget(self)
        self.layout.addWidget(self.config_list)

        # Populate the list with available .json files
        configs = [f for f in os.listdir(CONFIG_PATH) if f.endswith(".json")]
        self.config_list.addItems(configs)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_selected_config(self):
        selected = self.config_list.currentItem()
        return selected.text() if selected else None
