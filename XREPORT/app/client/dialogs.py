from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QVBoxLayout,
)

from XREPORT.app.utils.constants import CONFIG_PATH


###############################################################################
class SaveConfigDialog(QDialog):
    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Configuration As")
        self.dialog_layout = QVBoxLayout(self)

        self.label = QLabel("Enter a name for your configuration:", self)
        self.dialog_layout.addWidget(self.label)

        self.name_edit = QLineEdit(self)
        self.dialog_layout.addWidget(self.name_edit)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.dialog_layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_name(self) -> str:
        return self.name_edit.text().strip()


###############################################################################
class LoadConfigDialog(QDialog):
    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Configuration")
        self.dialog_layout = QVBoxLayout(self)

        self.label = QLabel("Select a configuration:", self)
        self.dialog_layout.addWidget(self.label)

        self.config_list = QListWidget(self)
        self.dialog_layout.addWidget(self.config_list)

        # Populate the list with available .json files
        configs: list[str] = [f for f in os.listdir(CONFIG_PATH) if f.endswith(".json")]
        self.config_list.addItems(configs)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.dialog_layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_selected_config(self) -> str | None:
        selected = self.config_list.currentItem()
        return selected.text() if selected else None
