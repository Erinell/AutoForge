# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QLabel,
    QLayout, QListWidget, QListWidgetItem, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_main(object):
    def setupUi(self, main):
        if not main.objectName():
            main.setObjectName(u"main")
        main.resize(916, 658)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main.sizePolicy().hasHeightForWidth())
        main.setSizePolicy(sizePolicy)
        main.setAutoFillBackground(False)
        main.setAnimated(True)
        main.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QWidget(main)
        self.centralwidget.setObjectName(u"centralwidget")
        self.importImage = QPushButton(self.centralwidget)
        self.importImage.setObjectName(u"importImage")
        self.importImage.setGeometry(QRect(20, 20, 101, 24))
        self.importMaterials = QPushButton(self.centralwidget)
        self.importMaterials.setObjectName(u"importMaterials")
        self.importMaterials.setGeometry(QRect(130, 20, 101, 24))
        self.materialsList = QListWidget(self.centralwidget)
        self.materialsList.setObjectName(u"materialsList")
        self.materialsList.setGeometry(QRect(330, 180, 256, 192))
        self.materialsList.setFrameShape(QFrame.Shape.StyledPanel)
        self.materialsList.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed)
        self.materialsList.setProperty(u"showDropIndicator", False)
        self.materialsList.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.materialsList.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.runButton = QPushButton(self.centralwidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(700, 20, 80, 24))
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(300, 0, 251, 131))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.iterationLabel = QLabel(self.verticalLayoutWidget)
        self.iterationLabel.setObjectName(u"iterationLabel")
        font = QFont()
        font.setPointSize(12)
        self.iterationLabel.setFont(font)
        self.iterationLabel.setTextFormat(Qt.TextFormat.PlainText)

        self.verticalLayout.addWidget(self.iterationLabel)

        self.lossLabel = QLabel(self.verticalLayoutWidget)
        self.lossLabel.setObjectName(u"lossLabel")
        self.lossLabel.setFont(font)
        self.lossLabel.setTextFormat(Qt.TextFormat.PlainText)

        self.verticalLayout.addWidget(self.lossLabel)

        self.bestlossLabel = QLabel(self.verticalLayoutWidget)
        self.bestlossLabel.setObjectName(u"bestlossLabel")
        self.bestlossLabel.setFont(font)
        self.bestlossLabel.setTextFormat(Qt.TextFormat.PlainText)

        self.verticalLayout.addWidget(self.bestlossLabel)

        self.tauLabel = QLabel(self.verticalLayoutWidget)
        self.tauLabel.setObjectName(u"tauLabel")
        self.tauLabel.setFont(font)
        self.tauLabel.setTextFormat(Qt.TextFormat.PlainText)

        self.verticalLayout.addWidget(self.tauLabel)

        self.layersLabel = QLabel(self.verticalLayoutWidget)
        self.layersLabel.setObjectName(u"layersLabel")
        self.layersLabel.setFont(font)
        self.layersLabel.setTextFormat(Qt.TextFormat.PlainText)

        self.verticalLayout.addWidget(self.layersLabel)

        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(10, 100, 241, 171))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.verticalLayoutWidget_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_2)

        self.targetPic = QLabel(self.verticalLayoutWidget_2)
        self.targetPic.setObjectName(u"targetPic")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.targetPic.sizePolicy().hasHeightForWidth())
        self.targetPic.setSizePolicy(sizePolicy2)
        self.targetPic.setMinimumSize(QSize(0, 0))
        self.targetPic.setFrameShape(QFrame.Shape.StyledPanel)
        self.targetPic.setFrameShadow(QFrame.Shadow.Plain)
        self.targetPic.setLineWidth(0)
        self.targetPic.setTextFormat(Qt.TextFormat.PlainText)
        self.targetPic.setScaledContents(True)

        self.verticalLayout_2.addWidget(self.targetPic)

        self.verticalLayoutWidget_3 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(10, 280, 241, 171))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.verticalLayoutWidget_3)
        self.label_3.setObjectName(u"label_3")
        sizePolicy1.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy1)
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_3)

        self.currentCompPic = QLabel(self.verticalLayoutWidget_3)
        self.currentCompPic.setObjectName(u"currentCompPic")
        sizePolicy2.setHeightForWidth(self.currentCompPic.sizePolicy().hasHeightForWidth())
        self.currentCompPic.setSizePolicy(sizePolicy2)
        self.currentCompPic.setMinimumSize(QSize(0, 0))
        self.currentCompPic.setFrameShape(QFrame.Shape.StyledPanel)
        self.currentCompPic.setFrameShadow(QFrame.Shadow.Plain)
        self.currentCompPic.setLineWidth(0)
        self.currentCompPic.setTextFormat(Qt.TextFormat.PlainText)
        self.currentCompPic.setScaledContents(True)

        self.verticalLayout_3.addWidget(self.currentCompPic)

        self.verticalLayoutWidget_4 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(10, 460, 241, 171))
        self.verticalLayout_4 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.verticalLayoutWidget_4)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)
        self.label_4.setFont(font)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_4)

        self.bestCompPic = QLabel(self.verticalLayoutWidget_4)
        self.bestCompPic.setObjectName(u"bestCompPic")
        sizePolicy2.setHeightForWidth(self.bestCompPic.sizePolicy().hasHeightForWidth())
        self.bestCompPic.setSizePolicy(sizePolicy2)
        self.bestCompPic.setMinimumSize(QSize(0, 0))
        self.bestCompPic.setFrameShape(QFrame.Shape.StyledPanel)
        self.bestCompPic.setFrameShadow(QFrame.Shadow.Plain)
        self.bestCompPic.setLineWidth(0)
        self.bestCompPic.setTextFormat(Qt.TextFormat.PlainText)
        self.bestCompPic.setScaledContents(True)

        self.verticalLayout_4.addWidget(self.bestCompPic)

        self.verticalLayoutWidget_5 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(360, 400, 241, 171))
        self.verticalLayout_6 = QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.verticalLayoutWidget_5)
        self.label_5.setObjectName(u"label_5")
        sizePolicy1.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy1)
        self.label_5.setFont(font)
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_5)

        self.heightmapPic = QLabel(self.verticalLayoutWidget_5)
        self.heightmapPic.setObjectName(u"heightmapPic")
        sizePolicy2.setHeightForWidth(self.heightmapPic.sizePolicy().hasHeightForWidth())
        self.heightmapPic.setSizePolicy(sizePolicy2)
        self.heightmapPic.setMinimumSize(QSize(0, 0))
        self.heightmapPic.setFrameShape(QFrame.Shape.StyledPanel)
        self.heightmapPic.setFrameShadow(QFrame.Shadow.Plain)
        self.heightmapPic.setLineWidth(0)
        self.heightmapPic.setTextFormat(Qt.TextFormat.PlainText)
        self.heightmapPic.setScaledContents(True)

        self.verticalLayout_6.addWidget(self.heightmapPic)

        main.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(main)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 916, 21))
        main.setMenuBar(self.menubar)

        self.retranslateUi(main)

        QMetaObject.connectSlotsByName(main)
    # setupUi

    def retranslateUi(self, main):
        main.setWindowTitle(QCoreApplication.translate("main", u"AutoForge", None))
        self.importImage.setText(QCoreApplication.translate("main", u"Import image", None))
        self.importMaterials.setText(QCoreApplication.translate("main", u"Import materials", None))
        self.runButton.setText(QCoreApplication.translate("main", u"Run", None))
        self.iterationLabel.setText(QCoreApplication.translate("main", u"0 / 0", None))
        self.lossLabel.setText(QCoreApplication.translate("main", u"Loss: 0.0", None))
        self.bestlossLabel.setText(QCoreApplication.translate("main", u"Best loss: 0.0", None))
        self.tauLabel.setText(QCoreApplication.translate("main", u"Tau: 0.0", None))
        self.layersLabel.setText(QCoreApplication.translate("main", u"Highest layer: 0mm", None))
        self.label_2.setText(QCoreApplication.translate("main", u"Target", None))
        self.targetPic.setText("")
        self.label_3.setText(QCoreApplication.translate("main", u"Current composite", None))
        self.currentCompPic.setText("")
        self.label_4.setText(QCoreApplication.translate("main", u"Best composite", None))
        self.bestCompPic.setText("")
        self.label_5.setText(QCoreApplication.translate("main", u"Height map", None))
        self.heightmapPic.setText("")
    # retranslateUi

