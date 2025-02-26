import sys
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem
from ui_form import Ui_main
from src.AutoForge import AutoForge, Callback
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

class main(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_main()
        self.ui.setupUi(self)
        self.setFixedSize(torch.Size(self.width(), self.height()))
        
        self.autoForge = AutoForge(self.ui)
        
        self.ui.importImage.clicked.connect(self.on_importImage_clicked)
        self.ui.importMaterials.clicked.connect(self.on_importMaterials_clicked)
        self.ui.runButton.clicked.connect(self.compute)
        
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)
    
    def compute(self):
        self.autoForge.update_signal.connect(self.on_update)
        self.autoForge.start()
    
    def on_update(self, data: Callback):
        self.ui.lossLabel.setText(f"Loss: {data.loss}")
        self.ui.bestlossLabel.setText(f"Best loss: {data.best_loss}")
        self.ui.tauLabel.setText(f"tau: {data.tau_height}")
        self.ui.iterationLabel.setText(f"{data.iteration} / {data.iterations}")
        self.ui.layersLabel.setText(f"Highest layer: {data.highest_layer:.2f}mm")
        self.ui.currentCompPic.setPixmap(QPixmap.fromImage(data.comp_im))
        self.ui.bestCompPic.setPixmap(QPixmap.fromImage(data.best_comp_im))
        self.ui.heightmapPic.setPixmap(QPixmap.fromImage(data.height_map_im))
        self.ui.discComPic.setPixmap(QPixmap.fromImage(data.disc_comp_im))
        
        
    def on_importImage_clicked(self):
        input_image = QFileDialog.getOpenFileUrl(self, "Open Image", "", "Image Files (*.jpg)")
        self.autoForge.input_image = input_image[0].toLocalFile()
        if self.autoForge.input_image:
            self.autoForge.load_target_image()
        
    def on_importMaterials_clicked(self):
        csv_file = QFileDialog.getOpenFileUrl(self, "Open CSV", "", "CSV Files (*.csv)")
        self.autoForge.csv_file = csv_file[0].toLocalFile()
        self.autoForge.load_materials()
        
        self.ui.materialsList.clear()
        for sheet in self.autoForge.material_names:
            item = QListWidgetItem(sheet)
            self.ui.materialsList.addItem(item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = main()
    widget.show()
    sys.exit(app.exec())
