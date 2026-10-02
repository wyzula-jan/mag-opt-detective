import sys
import os

# essential modules import to run app witg .ui #TODO at some point move to PyQt6
from PyQt5.QtWidgets import QMainWindow, QApplication#, QDesktopWidget #, QShortcut# essential
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QKeySequence, QColor
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import QProcess #, Qt
# App modules
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QFileDialog #, QWidget, QListWidget
import pyqtgraph as pg

from matplotlib.pyplot import cm

# from brukeropusreader import read_file
from scipy.interpolate import interp1d

#My custom Packages
from JW_cl4 import Measurement, Treatment, find_nearest
from JW_Qt2 import TableModel, ListBoxWidget
#Other Widnget Windows

# from Extractor_V1 import PointExtractor as PE

# for ui load
path = os.path.dirname(os.path.abspath(__file__))
qtCreatorFile = 'MOD_V4p8.ui'

Ui_MainWindow, QtBaseClass = uic.loadUiType(path + '/' + qtCreatorFile)



# class SliderOG(QWidget): #TODO Work in progress
#     def __init__(self,UIslider,val_label, minimum, maximum, parent=None):
#         super(Slider, self).__init__(parent=parent)
#         '''UI sliders'''
#         self.label = val_labels
#
#         self.slider = UIslider
#         self.slider.setValue(float(val_label.text()))
#         '''Values sets'''
#         self.minimum = minimum
#         self.maximum = maximum
#         self.x = val_label
#         self.slider.valueChanged.connect(self.setLabelValue)
#
#         self.setLabelValue(self.slider.value())
#
#     def setLabelValue(self, value):
#         self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
#         self.maximum - self.minimum)
#
#         self.label.setText("{0:.4g}".format(self.x))

# class Slider(QWidget):
#     def __init__(self,UIslider,val_label):#, minimum, maximum, testLabel):
#         self.slider = UIslider
#         self.label = val_label
#
#         self.slider.setRange(0,100)
#         self.slider.setPageStep(5)
#         self.slider.setFocusPolicy(Qt.NoFocus)
#
#         self.slider.valueChanged.connect(self.update)
#
#         # self.setLabelValue(self.slider.value())
#
#     def update(self,value):
#         self.label.setText(str(value))




# Main UI Class
class CM(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(CM, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Init Variables
        self.Zero_Sam = None
        self.Zero_Ref = None
        self.Fnames_Sam = None
        self.Fnames_Ref = None

        # Init for Scatter point extraction #TODO can be optimalized
        self.ClearPoints = False

        self.Processed = {}
        # # for loading optimalization
        # self.DataState = 0

        # #slider test
        # self.TestSlider = Slider(self.ui.testSlider,self.ui.testLabel)


        # Input LineEdits -> Enable/Disable according to the mode
        self.ui.comboFormat.currentIndexChanged.connect(self.Valid)

        '''Restriction for input - only float'''
        self.dv = QtGui.QDoubleValidator(0.0, 50.0, 4)
        for lineEdits in (self.ui.lineEdit_BStart, self.ui.lineEdit_BStep, self.ui.lineEdit_BEnd,
                          self.ui.lineEdit_BStart_2, self.ui.lineEdit_BStep_2, self.ui.lineEdit_BEnd_2,
                          self.ui.lineEdit_Bmin, self.ui.lineEdit_Bmax, self.ui.lineEdit_Emin, self.ui.lineEdit_Emax,
                          self.ui.lineEdit_Imin, self.ui.lineEdit_Imax, self.ui.lineEdit_BScorMin, self.ui.lineEdit_BScorMax):
            lineEdits.setValidator(self.dv)

        #No special Characters and space #TODO upgrade to pyqt6
        # self.expresion = QtCore.QRegExp("[a-z-A-Z_]+[0-9]")
        # self.ui.lineEdit_ColName.setValidator(QtGui.QRegExpValidator(self.expresion))

        ##############################
        # Button Assigment ###########
        ##############################

        # Main Top Buttons
        self.ui.ButtonTest.clicked.connect(self.Test)
        # self.ui.ButtonNew.clicked.connect(self.Start_NWindow)
        self.ui.ButtonCenterWindow.clicked.connect(self.location_on_the_screen)
        # self.ui.ButtonProcessed.clicked.connect(self.loadProcessedData)

        # Load Buttons
        self.ui.button_0Ta0T.clicked.connect(lambda: self.Load_Multiple(self.ui.listWidget_ZeroField, self.Zero_Sam))
        self.ui.button_field.clicked.connect(lambda: self.Load_Multiple(self.ui.listWidget_Field, self.Fnames_Sam))
        self.ui.button_0Ta0T_R.clicked.connect(lambda: self.Load_Multiple(self.ui.listWidget_ZeroField_R, self.Zero_Ref))
        self.ui.button_field_R.clicked.connect(lambda: self.Load_Multiple(self.ui.listWidget_Field_R, self.Fnames_Ref))
        # self.ui.buttonLoadProcessed.clicked.connect(self.loadProcessedData)

        # Plot Buttons
        # clear the glw before new plot:
        # Main process button
        self.ui.ButtonPlot.clicked.connect(self.ProcessData)
        # self.ui.ButtonPlot.clicked.connect(self.PostProcess)
        self.ui.ButtonPlot.clicked.connect(self.ui.glw.clear)
        self.ui.ButtonPlot.clicked.connect(self.ui.glw_2.clear)
        self.ui.ButtonPlot.clicked.connect(self.ui.glw_3.clear)
        # self.ui.ButtonPlot.clicked.connect(self.Test)
        self.ui.ButtonPlot.clicked.connect(self.PlotReference)
        self.ui.ButtonPlot.clicked.connect(self.PlotData)

        # Processed Buttons
        self.ui.ButtonPlotIndex.clicked.connect(self.PlotByIndex)
        self.ui.ButtonMergeE.clicked.connect(self.IndexMergeE)
        self.ui.ButtonSaveToIndex.clicked.connect(self.saveToIndex)

        # Point Extractor Buttons
        self.ui.ButtonDropCol.clicked.connect(self.dropCol)


        # if self.ui.checkBox_RefCorr.isChecked() or self.ui.checkBox_RefData.isChecked():
        # self.ui.ButtonPlot.clicked.connect(self.PlotReference)

        for PlotButtons in (self.ui.radio_Rat, self.ui.radio_Data, self.ui.radio_AVR,
                            self.ui.radio_Diff, self.ui.radio_DerNo, self.ui.radio_Der1st, self.ui.radio_Der2nd,
                            self.ui.ButtonReplot,self.ui.ButtonPlotIndex,self.ui.ButtonMergeE):
            PlotButtons.clicked.connect(self.ui.glw.clear)
            PlotButtons.clicked.connect(self.ui.glw_2.clear)
            PlotButtons.clicked.connect(self.PlotData)
            PlotButtons.clicked.connect(self.plotExtraction)

        # Reference plot button !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for PlotRefButtons in (self.ui.radioRefPlotRat, self.ui.radioRefPlotData):
            PlotRefButtons.clicked.connect(self.ui.glw_3.clear)
            PlotRefButtons.clicked.connect(self.PlotReference)

        # Processed Tab Buttons
        self.ui.ButtonLoadIndex.clicked.connect(self.loadProccessedCsv)

        #Widget Class to accept drag and drop

        self.ui.listWidget_ZeroField = ListBoxWidget(self.ui.listWidget_ZeroField)
        self.ui.listWidget_Field = ListBoxWidget(self.ui.listWidget_Field)
        self.ui.listWidget_ZeroField_R = ListBoxWidget(self.ui.listWidget_ZeroField_R)
        self.ui.listWidget_Field_R = ListBoxWidget(self.ui.listWidget_Field_R)

        # self.ui.pushButton_Test.clicked.connect(self.Test)

        #Export/Import Buttons
        self.ui.ButtonExport.clicked.connect(self.Export_csv)

        #Sliders
        # self.vf1 = SliderOG(self.ui.vf1Slider, self.ui.lineEdit_vf1,0.5,1.5)#,self.ui.testLabel)

        #Change the Tools page by combo box
        self.ui.comboTools.currentTextChanged.connect(self.switchStack)

        ##################################################
        # Keyboard shortcuts##############################
        ##################################################

        # Data Shortcut
        self.ui.ButtonPlot.setShortcut('Ctrl+F')
        self.ui.ButtonExport.setShortcut('Ctrl+E')

        ## #TODO all shortcuts to PyQt6
        # self.Load_0T_Sam_SH = QShortcut(QKeySequence('Ctrl+Q'), self)
        # self.Load_0T_Sam_SH.activated.connect(lambda: self.Load_Multiple(self.ui.listWidget_ZeroField, self.Zero_Sam))
        #
        # self.Load_Field_Sam_SH = QShortcut(QKeySequence('Ctrl+W'), self)
        # self.Load_Field_Sam_SH.activated.connect(lambda: self.Load_Multiple(self.ui.listWidget_Field, self.Fnames_Sam))
        #
        # self.Load_0T_Ref_SH = QShortcut(QKeySequence('Ctrl+A'), self)
        # self.Load_0T_Ref_SH.activated.connect(lambda: self.Load_Multiple(self.ui.listWidget_ZeroField_R, self.Zero_Ref))
        # # self.Load_0T_Ref_SH.activated.connect(self.RefCheck)
        #
        # self.Load_Field_Ref_SH = QShortcut(QKeySequence('Ctrl+S'), self)
        # self.Load_Field_Ref_SH.activated.connect(lambda: self.Load_Multiple(self.ui.listWidget_Field_R, self.Fnames_Ref))
        # self.Load_Field_Ref_SH.activated.connect(self.RefCheck)

        # QShortcut(QKeySequence('Ctrl+Q'), self).connect(self.Load_0T_Sam)
        # self.ui.checkBox_RefCorr.setChecked().setShortcut('Ctrl+S')

        # plot shortcuts
        self.ui.radio_Rat.setShortcut('Ctrl+1')
        self.ui.radio_Data.setShortcut('Ctrl+2')
        self.ui.radio_AVR.setShortcut('Ctrl+3')
        # self.ui.radio_Diff.setShortcut('Ctrl+4')

        self.ui.radio_DerNo.setShortcut('Alt+1')
        self.ui.radio_Der1st.setShortcut('Alt+2')
        self.ui.radio_Der2nd.setShortcut('Alt+3')

        ##################################################
                       # Recorder
        ##################################################

        self.ui.ButtonPlotAllPoints.clicked.connect(self.plotExtraction)
        self.ui.Button_PointExport.clicked.connect(self.ExportPoints)

        self.ui.ButtonLoadPoints.clicked.connect(self.LoadPoints)

        ##################################################
                       # Test Buttons
        ##################################################
        # self.ui.Button_RowTest.clicked.connect(self.TableTest)
        # self.ui.ButtonAdd1.clicked.connect(self.Add1)
        self.ui.ButtonTableTest.clicked.connect(self.IndexTest)

        ##################################################
        # Widget Window Assignment #######################
        #############Not USed so far###############

        # self.PE = PE()
        #
        # self.ui.ButtonExtractor.clicked.connect(self.PE.show)



    ###############################################################
    ################# Class functions #############################
    ###############################################################
    def switchStack(self):
        self.ui.stkW_Tools.setCurrentIndex(self.ui.comboTools.currentIndex())

    ########## Validator for enable/disable input #################
    def Valid(self):
        print(self.ui.comboFormat.currentText())
        if self.ui.comboFormat.currentIndex() == 0:
            self.DisableInput_Sam()
            self.DisableInput_Ref()
            print('Sam DIS')
        elif self.ui.comboFormat.currentIndex() == 1:
            self.EnableInput_Sam()
            self.EnableInput_Ref()
            print('Sam EN')
        elif self.ui.comboFormat.currentIndex() == 2:
            self.EnableInput_Sam()
            self.EnableInput_Ref()
            print('Sam OPUS')
        else:
            print('No restriction')

    def DisableInput_Sam(self):
        self.ui.lineEdit_BEnd.setDisabled(True)
        self.ui.lineEdit_BStart.setDisabled(True)
        self.ui.lineEdit_BStep.setDisabled(True)

    def EnableInput_Sam(self):
        self.ui.lineEdit_BEnd.setDisabled(False)
        self.ui.lineEdit_BStart.setDisabled(False)
        self.ui.lineEdit_BStep.setDisabled(False)

    def DisableInput_Ref(self):
        self.ui.lineEdit_BEnd_2.setDisabled(True)
        self.ui.lineEdit_BStart_2.setDisabled(True)
        self.ui.lineEdit_BStep_2.setDisabled(True)

    def EnableInput_Ref(self):
        self.ui.lineEdit_BEnd_2.setDisabled(False)
        self.ui.lineEdit_BStart_2.setDisabled(False)
        self.ui.lineEdit_BStep_2.setDisabled(False)

    def DisableBLimit(self):
        self.ui.lineEdit_Bmin.setDisabled(True)
        self.ui.lineEdit_Bmax.setDisabled(True)

    def EnableBLimit(self):
        self.ui.lineEdit_Bmin.setDisabled(False)
        self.ui.lineEdit_Bmax.setDisabled(False)



    def Test(self):
        items = []
        listUI = self.ui.listWidget_ZeroField
        for x in range(listUI.count()):
            items.append(listUI.item(x).text())

        print('####### Script Path #######')
        print(sys.path[0])
        print('####### Python Interpreter Path #######')
        print(sys.path[1])
        print('####### Version #######')
        print(sys.version)
        print('####### Fnames #######')
        print(items)
        print('###### Links #########')
        print(self.ui.listWidget_ZeroField.links)

    ##########################################################
    ########## Data Processing ###############################
    ##########################################################

    def ProcessData(self):

        self.ui.ConsoleOutput.append(60*'#')

        #paths from list widgets
        self.Zero_Sam = self.ui.listWidget_ZeroField.links
        self.Fnames_Sam = self.ui.listWidget_Field.links

        self.Zero_Ref = self.ui.listWidget_ZeroField_R.links
        self.Fnames_Ref = self.ui.listWidget_Field_R.links

        #Derivative -> Energy (index = 0) vs Field (index = 1)
        derAxes = self.ui.comboDerivative.currentIndex()

        # Setting Limits
        if self.ui.radio_ElimCut.isChecked():
            self.E_Lo = float(self.ui.lineEdit_Emin.text())
            self.E_Hi = float(self.ui.lineEdit_Emax.text())
        else:
            self.E_Lo = None
            self.E_Hi = None

            # Loading
        if self.ui.comboFormat.currentIndex() == 0:  # OPUS Macro ASCII Regime
            self.meas = Measurement(ZeroList=self.Zero_Sam, \
                                    FieldList=self.Fnames_Sam, \
                                    units=self.ui.comboUnit.currentText(), \
                                    B=None, limits=(self.E_Lo, self.E_Hi))
            self.xlabels = self.meas.B
            self.B = self.xlabels

            if self.ui.checkBox_RefCorr.isChecked():
                self.ref = Measurement(ZeroList=self.Zero_Ref, \
                                       FieldList=self.Fnames_Ref, \
                                       units=self.ui.comboUnit.currentText(), \
                                       B=None, limits=(self.E_Lo, self.E_Hi))
                self.B_ref = self.ref.B

        elif self.ui.comboFormat.currentIndex() == 1:  # ASCII - Custom Field
            self.B = np.arange(float(self.ui.lineEdit_BStart.text()),
                               float(self.ui.lineEdit_BEnd.text()) + float(self.ui.lineEdit_BStep.text()),
                               float(self.ui.lineEdit_BStep.text()))
            self.meas = Measurement(ZeroList=self.Zero_Sam, \
                                    FieldList=self.Fnames_Sam, \
                                    units=self.ui.comboUnit.currentText(), \
                                    B=self.B, limits=(self.E_Lo, self.E_Hi))
            self.xlabels = self.B

            if self.ui.checkBox_RefCorr.isChecked():
                self.B_ref = np.arange(float(self.ui.lineEdit_BStart_2.text()),
                                       float(self.ui.lineEdit_BEnd_2.text()) + float(self.ui.lineEdit_BStep_2.text()),
                                       float(self.ui.lineEdit_BStep_2.text()))
                self.meas = Measurement(ZeroList=self.Zero_Ref, \
                                        FieldList=self.Fnames_Ref, \
                                        units=self.ui.comboUnit.currentText(), \
                                        B=self.B_ref, limits=(self.E_Lo, self.E_Hi))

        elif self.ui.comboFormat.currentIndex() == 2:  # OPUS - Custom Field
            self.B = np.arange(float(self.ui.lineEdit_BStart.text()),
                               float(self.ui.lineEdit_BEnd.text()) + float(self.ui.lineEdit_BStep.text()),
                               float(self.ui.lineEdit_BStep.text()))
            self.meas = Measurement(ZeroList=self.Zero_Sam, \
                                    FieldList=self.Fnames_Sam, \
                                    units=self.ui.comboUnit.currentText(), \
                                    B=self.B, OPUS=True, limits=(self.E_Lo, self.E_Hi))
            self.xlabels = self.B

            if self.ui.checkBox_RefCorr.isChecked():
                self.B_ref = np.arange(float(self.ui.lineEdit_BStart_2.text()),
                                       float(self.ui.lineEdit_BEnd_2.text()) + float(self.ui.lineEdit_BStep_2.text()),
                                       float(self.ui.lineEdit_BStep_2.text()))
                self.meas = Measurement(ZeroList=self.Zero_Ref, \
                                        FieldList=self.Fnames_Ref, \
                                        units=self.ui.comboUnit.currentText(), \
                                        B=self.B_ref, OPUS=True, limits=(self.E_Lo, self.E_Hi))

        if self.ui.checkBox_RefCorr.isChecked(): #TODO not sure if elif or if -> has to be debugged
            self.Data_Ref = Treatment.Interpolate(data=self.ref.Field(), B0=self.B_ref, B=self.B)
            self.Ratio_Ref = Treatment.Interpolate(data=self.ref.Ratio(), B0=self.B_ref, B=self.B)

            if self.ui.checkBox_RefSG.isChecked():
                self.Ratio_Ref = Treatment.SG_smooth(data=self.Ratio_Ref, window=int(self.ui.lineEdit_SG_Window.text()),
                                                     poly=int(self.ui.lineEdit_SG_Poly.text()))
                self.Data_Ref = Treatment.SG_smooth(data=self.Data_Ref, window=int(self.ui.lineEdit_SG_Window.text()),
                                                    poly=int(self.ui.lineEdit_SG_Poly.text()))

            self.Data = self.meas.Field() / self.Data_Ref
            self.Ratio = self.meas.Ratio() / self.Ratio_Ref
            self.Ratio_AVR = self.Data.div(self.Data.mean(axis=1), axis=0)

            # Triger postprocessing check and generate table for points
            self.PostProcess()
            self.InitTable()

            # Derivatives:
            self.Data_Der1st = Treatment.derivative(self.Data, axis = derAxes)
            self.Data_Der2nd = Treatment.derivative(self.Data_Der1st, axis = derAxes)

            self.Ratio_Der1st = Treatment.derivative(self.Ratio, axis = derAxes)
            self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st, axis = derAxes)

            self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR, axis = derAxes)
            self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st, axis = derAxes)

            self.ylabels = self.Data.index.values


        elif self.ui.checkBox_RefData.isChecked():
            self.Data_Ref = self.meas.Field()
            self.Ratio_Ref = self.meas.Ratio()
            # self.ui.checkBox_RefSG.setChecked(True)
            if self.ui.checkBox_RefSG.isChecked():
                self.Ratio_Ref = Treatment.SG_smooth(data=self.Ratio_Ref, window=int(self.ui.lineEdit_SG_Window.text()),
                                                     poly=int(self.ui.lineEdit_SG_Poly.text()))
                self.Data_Ref = Treatment.SG_smooth(data=self.Data_Ref, window=int(self.ui.lineEdit_SG_Window.text()),
                                                    poly=int(self.ui.lineEdit_SG_Poly.text()))

            self.Data = self.meas.Field() / self.Data_Ref
            self.Ratio = self.meas.Ratio() / self.Ratio_Ref
            self.Ratio_AVR = self.Data.div(self.Data.mean(axis=1), axis=0)

            # Triger postprocessing check and generate table for points
            self.PostProcess()
            self.InitTable()

            # Derivatives:
            self.Data_Der1st = Treatment.derivative(self.Data, axis = derAxes)
            self.Data_Der2nd = Treatment.derivative(self.Data_Der1st, axis = derAxes)

            self.Ratio_Der1st = Treatment.derivative(self.Ratio, axis = derAxes)
            self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st, axis = derAxes)

            self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR, axis = derAxes)
            self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st, axis = derAxes)

            self.ylabels = self.Data.index.values


        else:
            self.Data = self.meas.Field()
            self.Ratio = self.meas.Ratio()
            self.Ratio_AVR = self.Data.div(self.Data.mean(axis=1), axis=0)

            # Triger postprocessing check and generate table for points
            self.PostProcess()
            self.InitTable()

            # Derivatives:
            self.Data_Der1st = Treatment.derivative(self.Data, axis = derAxes)
            self.Data_Der2nd = Treatment.derivative(self.Data_Der1st, axis = derAxes)

            self.Ratio_Der1st = Treatment.derivative(self.Ratio, axis = derAxes)
            self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st, axis = derAxes)

            self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR, axis = derAxes)
            self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st, axis = derAxes)

            self.ylabels = self.Data.index.values

    #PostProcessing of the data
    def PostProcess(self):
        #Base line correction - normalization of the region to unity
        if self.ui.radio_BScor_True.isChecked():
            self.Ratio = Treatment.BS_correct(data = self.Ratio, region=[float(self.ui.lineEdit_BScorMin.text()),float(self.ui.lineEdit_BScorMax.text())])
            self.Ratio_AVR = Treatment.BS_correct(data = self.Ratio_AVR, region=[float(self.ui.lineEdit_BScorMin.text()),float(self.ui.lineEdit_BScorMax.text())])

            print(f'Baseline Corrected in range {self.ui.lineEdit_BScorMin.text()} - {self.ui.lineEdit_BScorMax.text()} {self.ui.comboUnit.currentText()}.')
            self.ui.ConsoleOutput.append(f'Baseline Corrected in range {self.ui.lineEdit_BScorMin.text()} - {self.ui.lineEdit_BScorMax.text()} {self.ui.comboUnit.currentText()}.')
        else:
            print('No postprocessing.')
            self.ui.ConsoleOutput.append('No postprocessing.')


    #############################################
    ############### Plotting Data ###############
    #############################################

    def PlotReference(self):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if self.ui.checkBox_RefCorr.isChecked() or self.ui.checkBox_RefData.isChecked():
            if self.ui.radioRefPlotRat.isChecked():
                self.plotCM(glw=self.ui.glw_3, data=self.Ratio_Ref.values.T, \
                            Hlimit=[float(self.ui.lineEdit_Imin.text()), float(self.ui.lineEdit_Imax.text())], \
                            cmap='magma')

            elif self.ui.radioRefPlotData.isChecked():
                self.plotCM(glw=self.ui.glw_3, data=self.Data_Ref.values.T, \
                            Hlimit=[float(self.ui.lineEdit_Datamin.text()), float(self.ui.lineEdit_Datamax.text())], \
                            cmap='magma')
        else:
            print('No Reference')
            self.ui.ConsoleOutput.append('No Reference')

    def PlotData(self):

        if self.ui.radio_Rat.isChecked():  # Plot Ratio
            if self.ui.radio_DerNo.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio.values.T, \
                            Hlimit=[float(self.ui.lineEdit_Imin.text()), float(self.ui.lineEdit_Imax.text())], \
                            cmap='magma')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio.values.T)

            elif self.ui.radio_Der1st.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio_Der1st.values.T, \
                            Hlimit=[float(self.ui.lineEdit_1stDermin.text()), float(self.ui.lineEdit_1stDermax.text())], \
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio_Der1st.values.T)

            elif self.ui.radio_Der2nd.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio_Der2nd.values.T, \
                            Hlimit=[float(self.ui.lineEdit_2ndDermin.text()), float(self.ui.lineEdit_2ndDermax.text())], \
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio_Der2nd.values.T)

        elif self.ui.radio_Data.isChecked():  # Plot Data
            if self.ui.radio_DerNo.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Data.values.T, \
                            Hlimit=[float(self.ui.lineEdit_Datamin.text()), float(self.ui.lineEdit_Datamax.text())], \
                            cmap='magma')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Data.values.T)

            elif self.ui.radio_Der1st.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Data_Der1st.values.T, \
                            Hlimit=[float(self.ui.lineEdit_1stDermin.text()), float(self.ui.lineEdit_1stDermax.text())], \
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Data_Der1st.values.T)

            elif self.ui.radio_Der2nd.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Data_Der2nd.values.T, \
                            Hlimit=[float(self.ui.lineEdit_2ndDermin.text()), float(self.ui.lineEdit_2ndDermax.text())], \
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Data_Der2nd.values.T)

        elif self.ui.radio_AVR.isChecked():  # Plot Ratio AVR
            if self.ui.radio_DerNo.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio_AVR.values.T, \
                            Hlimit=[float(self.ui.lineEdit_Imin.text()), float(self.ui.lineEdit_Imax.text())], \
                            cmap='magma')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio_AVR.values.T)

            elif self.ui.radio_Der1st.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio_AVR_Der1st.values.T,
                            Hlimit=[float(self.ui.lineEdit_1stDermin.text()), float(self.ui.lineEdit_1stDermax.text())],
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio_AVR_Der1st.values.T)

            elif self.ui.radio_Der2nd.isChecked():
                self.plotCM(glw=self.ui.glw, data=self.Ratio_AVR_Der2nd.values.T,
                            Hlimit=[float(self.ui.lineEdit_2ndDermin.text()), float(self.ui.lineEdit_2ndDermax.text())],
                            cmap='grey')
                if self.ui.radioSTK_True.isChecked():
                    self.plotSTK(data=self.Ratio_AVR_Der2nd.values.T)

    def plotSTK(self, data):

        self.glw_STK = self.ui.glw_2
        self.data_STK = data
        self.glw_STK.show()

        self.label_STK = pg.LabelItem(justify='center')
        self.glw_STK.addItem(self.label_STK)

        self.glw_STK.nextRow()

        self.p1_STK = self.glw_STK.addPlot(1, 0)

        self.p1_STK.setLabel(axis='bottom', text=f'Energy ({self.ui.comboUnit.currentText()})')

        self.p1_STK.setLabel(axis='left', text='Intensity (A.U.)')

        if self.ui.radio_ElimCustom.isChecked():
            self.p1_STK.setRange(xRange=[float(self.ui.lineEdit_Emin.text()), float(self.ui.lineEdit_Emax.text())])
        if self.ui.radio_STKcustom.isChecked():
            self.p1_STK.setRange(
                yRange=[float(self.ui.lineEdit_ISTKmin.text()), float(self.ui.lineEdit_ISTKmax.text())])

        self.offset = float(self.ui.lineEdit_offset.text())

        for i in range(0, np.size(self.data_STK, axis=0)):
            self.p1_STK.plot(self.ylabels, self.data_STK[i, :] + i * self.offset)

        # crosshair
        self.vLine_STK = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_STK = pg.InfiniteLine(angle=0, movable=False)
        self.p1_STK.addItem(self.vLine_STK, ignoreBounds=True)
        self.p1_STK.addItem(self.hLine_STK, ignoreBounds=True)

        self.vb_STK = self.p1_STK.vb

        #Point extractor -> Scatter plot
        self.s1 = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='x', size=1)
        self.p1.addItem(self.s1)

        def mouseMoved(evt_STK):
            pos_STK = evt_STK[0]  ## using signal proxy turns original arguments into a tuple
            if self.p1_STK.sceneBoundingRect().contains(pos_STK):
                mousePoint_STK = self.vb_STK.mapSceneToView(pos_STK)
                self.label_STK.setText("x=%0.1f, y=%0.1f" % (mousePoint_STK.x(), mousePoint_STK.y()))
                self.vLine_STK.setPos(mousePoint_STK.x())
                self.hLine_STK.setPos(mousePoint_STK.y())

        self.proxy_STK = pg.SignalProxy(self.p1_STK.scene().sigMouseMoved, rateLimit=60,
                                        slot=mouseMoved)  # update crosshair position

    # %%    '''Plotting functions'''

    def plotCM(self, glw, data, Hlimit, cmap): #TODO change imageview to plotview
        self.glw = glw

        self.Plotdata = data

        self.glw.show()
        # label for position
        self.label = pg.LabelItem(justify='center')
        self.glw.addItem(self.label)

        self.glw.nextRow()
        self.p1 = self.glw.addPlot(1, 0)
        self.p1.setLabel(axis='bottom', text='Magnetic Field (T)')
        self.p1.setLabel(axis='left', text=f'Energy ({self.ui.comboUnit.currentText()})')

        if self.ui.radio_BlimSame.isChecked():
            if self.ui.comboFormat.currentIndex() == 0:
                    self.p1.setXRange(float(self.B[0]), float(self.B[-1]), padding=0)
            else:
                self.p1.setXRange(float(self.ui.lineEdit_BStart.text()), float(self.ui.lineEdit_BEnd.text()), padding=0)

        elif self.ui.radio_BlimCustom.isChecked():
            self.p1.setXRange(float(self.ui.lineEdit_Bmin.text()), float(self.ui.lineEdit_Bmax.text()), padding=0)

        if self.ui.radio_ElimCustom.isChecked():
            self.p1.setRange(yRange=[float(self.ui.lineEdit_Emin.text()), float(self.ui.lineEdit_Emax.text())])

        #Seting image and transformation of the axes
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)

        self.tr = QtGui.QTransform()
        self.tr.translate(self.xlabels[0], self.ylabels[0])
        self.tr.scale((self.xlabels[-1] - self.xlabels[0]) / len(self.xlabels),
                      (self.ylabels[-1] - self.ylabels[0]) / len(self.ylabels))
        self.img.setTransform(self.tr)

        self.img.updateImage(image=self.Plotdata, levels=(Hlimit[0], Hlimit[1]))
        self.img.dataTransform()


        # HISTOGRAM

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.gradient.loadPreset(cmap)
        if self.ui.radio_IlimCustom.isChecked():
            self.hist.setLevels(min=Hlimit[0], max=Hlimit[1])
            self.hist.setHistogramRange(Hlimit[0] - 0.1 * Hlimit[0], Hlimit[1] + 0.1 * Hlimit[1])
        self.glw.addItem(self.hist)

        # cross hair #NEW
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.p1.addItem(self.vLine, ignoreBounds=True)
        self.p1.addItem(self.hLine, ignoreBounds=True)

        self.vb = self.p1.vb

        #Mouse interactions

        #Scatter for points
        self.s1 = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='x', size=1)
        self.p1.addItem(self.s1)

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.p1.sceneBoundingRect().contains(pos):
                mousePoint = self.vb.mapSceneToView(pos)
                self.label.setText("x=%0.1f, y=%0.1f" % (mousePoint.x(), mousePoint.y()))
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(mousePoint.y())

        def onClick(event):
            items = self.p1.scene().items(event.scenePos())
            mousePoint = self.vb.mapSceneToView(event._scenePos)
            if self.ui.radio_Record.isChecked():
                # del(mousePoint.x(), mousePoint.y())
                self.ExtData.loc[find_nearest(self.B, mousePoint.x()), self.ui.lineEdit_ColName.text()] = mousePoint.y()
                self.model = TableModel(self.ExtData)
                self.table.setModel(self.model)

                self.s1.setData(self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].index.values
                                ,self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].values)#,
                print(mousePoint.x(), mousePoint.y())

            elif self.ui.radio_Remove.isChecked():
                self.ExtData.loc[find_nearest(self.B, mousePoint.x()), self.ui.lineEdit_ColName.text()] = np.NaN
                self.model = TableModel(self.ExtData)
                self.table.setModel(self.model)

                self.s1.setData(self.ExtData.loc[:, self.ui.lineEdit_ColName.text()].index.values
                                , self.ExtData.loc[:, self.ui.lineEdit_ColName.text()].values)  # ,
                print(mousePoint.x(), mousePoint.y())
            else:
                print('No record/remove mode')

        self.proxy = pg.SignalProxy(self.p1.scene().sigMouseMoved, rateLimit=60,
                                    slot=mouseMoved)  # update crosshair position

        self.p1.scene().sigMouseClicked.connect(onClick)

    ##########################################################
    ##########  Processed Data  ##############################
    ##########################################################
    def loadProccessedCsv(self):
        fname = QFileDialog.getOpenFileName()

        rowIndex = self.ui.spinBox_index_proc.value()

        self.Processed[rowIndex] = pd.read_csv(str(fname[0]), sep='\t', index_col=0)

        if self.ui.checkBox_AutoB.isChecked():
            self.B= np.array([float(self.Processed[rowIndex].columns[i][:-1])
                                     for i in range(0,self.Processed[rowIndex].columns.size)])

        self.ui.tableWidget_loaded.setItem(rowIndex,0,QTableWidgetItem(fname[0].split('/')[-1]))

        print(fname[0].split('/')[-1])

        # def setColortoRow(table, rowIndex, color):
        #     for j in range(table.columnCount()):
        #         # table.setItem(rowIndex, j, QTableWidgetItem())
        #         table.item(rowIndex, j).setBackground(color)
        #
        #
        # setColortoRow(self.ui.tableWidget_loaded, rowIndex, QtGui.QColor(125,125,125))

        # print(self.Processed)

    def saveToIndex(self):
        rowIndex = self.ui.spinBox_index_proc.value()

        self.Processed[rowIndex] = self.Ratio

        if self.ui.checkBox_AutoB.isChecked():
            self.B= np.array([float(self.Processed[rowIndex].columns[i][:-1])
                                     for i in range(0,self.Processed[rowIndex].columns.size)])

        self.ui.tableWidget_loaded.setItem(rowIndex,0,QTableWidgetItem(f'Saved {rowIndex}'))

    def IndexTest(self):

        print('#####Whole Thing:')
        print(self.Processed)

        print('##### Values:')
        print(self.Processed.values())

        print('##### Keys:')
        print(self.Processed.keys())

        emin = self.ui.tableWidget_loaded.item(0,1).text()
        emax = self.ui.tableWidget_loaded.item(0,2).text()

        eminINT = int(self.ui.tableWidget_loaded.item(0,1).text())
        emaxINT = int(emax)
        print('Emin - Emax')
        print(eminINT,emaxINT)
        print(type(eminINT),type(emaxINT))

    def PlotByIndex(self): #TODO Optimalization of all Ratio etc. could be done into one function

        self.ui.ConsoleOutput.append(60*'#')
        indexPlot = self.ui.spinBox_index_proc.value()

        self.ui.ConsoleOutput.append(f'Plotting {self.ui.tableWidget_loaded.item(indexPlot,0).text()}')

        if self.ui.checkBox_AutoE.isChecked():
            self.Data = self.Processed[indexPlot]
            self.Ratio = self.Processed[indexPlot]
            self.Ratio_AVR = self.Processed[indexPlot]
        else:
            Emin = int(self.ui.tableWidget_loaded.item(indexPlot,1).text())
            Emax = int(self.ui.tableWidget_loaded.item(indexPlot,2).text())
            self.Data = self.Processed[indexPlot].loc[Emin:Emax,:]
            self.Ratio = self.Processed[indexPlot].loc[Emin:Emax,:]
            self.Ratio_AVR = self.Processed[indexPlot].loc[Emin:Emax,:]

        if self.ui.checkBox_AutoB.isChecked(): #TODO probably will not be needed, processed data always have field reference
            self.B = np.array([float(self.Ratio.columns[i][:-1])
                               for i in range(0, self.Ratio.columns.size)])
        else:
            B_start_p = self.ui.lineEdit_BStart.text()
            B_step_p = self.ui.lineEdit_BStep.text()
            B_end_p = self.ui.lineEdit_BEnd.text()

            self.B = np.arange(float(B_start_p), float(B_end_p) + float(B_step_p), float(B_step_p))

        self.xlabels = self.B

        # Triger postprocessing check and generate table for points
        self.PostProcess()
        self.InitTable()

        # Derivatives:
        self.Data_Der1st = Treatment.derivative(self.Data)
        self.Data_Der2nd = Treatment.derivative(self.Data_Der1st)

        self.Ratio_Der1st = Treatment.derivative(self.Ratio)
        self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st)

        self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR)
        self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st)

        self.ylabels = self.Data.index.values

#TODO process buttons for AVERAGE, MERGE E and B
    # def IndexAverage(self):
    def getRes(self,data, form=4): #TODO implement to separate package when working
        return round((data.index.values[-1] - data.index.values[0]) / np.size(data.index.values), 4)

    def InterpolateE(self,data, E0, E):
        Fint = interp1d(E0, data.T.values, fill_value='extrapolate')
        return pd.DataFrame(data=Fint(E).T, index=E,
                            columns=data.columns)

    def IndexMergeE(self): #TODO optimalize

        # b = True if True in np.isnan(np.array(a)) else False
        # print(b)

        Data = self.Processed

        # EnergyCheck = [self.ui.tableWidget_loaded.item(i,1).text() for i in Data] \ #TODO warning about energy
        #                + [self.ui.tableWidget_loaded.item(i,2).text() for i in Data]
        #
        # E_bool = True if True in np.isnan(np.array(EnergyCheck)) else False
        # print(E_bool)
        # # redColor = QColor(255, 0, 0)
        # if E_bool is True:
        #     ErrorMes = 'Energy limits are not set. Set limits for each dataset !'
        #     self.ui.ConsoleOutput.append(ErrorMes)

        self.Merged = pd.concat([Data[i].loc[
                                 int(self.ui.tableWidget_loaded.item(i,1).text()):
                                 int(self.ui.tableWidget_loaded.item(i,2).text())]
                                 for i in Data.keys()])

        #TODO REindex energy
        # index_og = self.Merged.index.values
        self.Merged = self.InterpolateE(self.Merged,
                                        E0 = self.Merged.index.values,
                                        E = np.arange(self.Merged.index.values[0],
                                                      self.Merged.index.values[-1]+self.getRes(self.Merged),
                                                      self.getRes(self.Merged)))

        self.Data = self.Merged
        self.Ratio = self.Merged
        self.Ratio_AVR = self.Merged

        self.xlabels = self.B

        # Derivatives:
        self.Data_Der1st = Treatment.derivative(self.Data)
        self.Data_Der2nd = Treatment.derivative(self.Data_Der1st)

        self.Ratio_Der1st = Treatment.derivative(self.Ratio)
        self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st)

        self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR)
        self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st)

        self.ylabels = self.Data.index.values

        #post process + Init Table for points
        self.ui.ConsoleOutput.append(60*'#')

        self.PostProcess()
        self.InitTable()

        self.ui.ConsoleOutput.append(f'{[self.ui.tableWidget_loaded.item(i,0).text() for i in Data]} merged')
        self.ui.ConsoleOutput.append(f'Energy scale interpolated to match different resolutions.')
    # def IndexMergeB(self):

    #######################################
    ################ TOOLS ################
    #######################################

    # def Extractor(self):
    #     print('Extractor Triggered')
    #     # self.glw = glw
    #     def mouseMoved(evt):
    #         pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    #         if self.p1.sceneBoundingRect().contains(pos):
    #             mousePoint = self.vb.mapSceneToView(pos)
    #             self.ui.label_Extractor.setText("[%0.1f, %0.1f]" % (mousePoint.x(), mousePoint.y()))
    #             self.vLine.setPos(mousePoint.x())
    #             self.hLine.setPos(mousePoint.y())
    #
    #     self.proxyE = pg.SignalProxy(self.p1.scene().sigMouseMoved, rateLimit=60,
    #                                 slot=mouseMoved)  # update crosshair position


    #######################################
    ########### Point Table ###############
    #######################################

    def InitTable(self):
        if self.ui.checkBox_InitTable.isChecked():
            init = np.empty(np.shape(self.B))
            init[:] = np.NaN
            self.table = self.ui.tableWidget_Extractor
            self.ExtData = pd.DataFrame(data=init, columns=[self.ui.lineEdit_ColName.text()], index = self.B)

            self.model = TableModel(self.ExtData)
            self.table.setModel(self.model)
            self.ui.checkBox_InitTable.setChecked(False)
            print('New point extraction table initialized.')
            self.ui.ConsoleOutput.append('New point extraction table initialized.')
        else:
            print('Point extraction table already initialized.')
            self.ui.ConsoleOutput.append('Point extraction table already initialized.')

    # def RecordTest(self):#TODO Implement the data to tableview with custom column
    # #TODO probably not needed as separate function
    #     if self.ui.checkBox_InitTable.isChecked(): # #TODO Dont care about this init
    #         self.InitTable()
    #
    #     self.ui.checkBox_InitTable.setChecked(False)
    #
    #     self.s1 = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='x', size=1)
    #     self.p1.addItem(self.s1)
    #
    #     def onClick(event):
    #         items = self.p1.scene().items(event.scenePos())
    #         mousePoint = self.vb.mapSceneToView(event._scenePos)
    #         self.ExtData.loc[find_nearest(self.B, mousePoint.x()), self.ui.lineEdit_ColName.text()] = mousePoint.y()
    #         self.model = TableModel(self.ExtData)
    #         self.table.setModel(self.model)
    #
    #         self.s1.setData(self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].index.values
    #                         ,self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].values)#,
    #         print(mousePoint.x(), mousePoint.y())

        # def RemoveClick(event):
        #     items = self.p1.scene().items(event.scenePos())
        #     mousePoint = self.vb.mapSceneToView(event._scenePos)
        #     self.ExtData.loc[find_nearest(self.B, mousePoint.x()), self.ui.lineEdit_ColName.text()] = np.nan
        #     self.model = TableModel(self.ExtData)
        #     self.table.setModel(self.model)
        #
        #     self.s1.setData(self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].index.values
        #                     ,self.ExtData.loc[:,self.ui.lineEdit_ColName.text()].values)#,
        #     print(mousePoint.x(), mousePoint.y())


    def plotExtraction(self):

        if self.ClearPoints is True:
            self.p1.removeItem(self.s2)

        self.s2 = pg.ScatterPlotItem(symbol='o', size=1)
        self.p1.addItem(self.s2)

        self.s2.clear() #TODO just thinking why this is here??

        colors = cm.jet(np.linspace(0, 1, np.size(self.ExtData.columns))) * 255

        for i in range(0,np.size(self.ExtData.columns)):
            self.s2.addPoints(self.ExtData.iloc[:,i].index.values,self.ExtData.iloc[:,i].values, pen = pg.mkPen(width=5, color= tuple(colors[i])))

        self.ClearPoints = True

    def dropCol(self):
        self.ExtData = self.ExtData.drop(columns=[self.ui.lineEdit_ColName.text()])
        self.model = TableModel(self.ExtData)
        self.table.setModel(self.model)

    def ExportPoints(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter='*.csv')

        if (name[0] == ''):
            pass
        else:
            self.ExtData.to_csv(name[0],sep='\t')

    def LoadPoints(self):

        fname = QFileDialog.getOpenFileName()

        self.table = self.ui.tableWidget_Extractor
        self.ExtData = pd.read_csv(str(fname[0]), sep='\t', index_col=0)

        self.model = TableModel(self.ExtData)
        self.table.setModel(self.model)


    ##############Plot Model#####################
    # def Dirac(self):
    #
    #     self.Dirac_update()
    #     self.vf1.slider.valueChanged.connect(self.Dirac_update)
    #
    #
    # def Dirac_update(self):
    #
    #     vf1 = self.vf1.x
    #     self.ui.lineEdit_vf1.setText(f'{self.vf:.2f}')


    ##############Export csv#####################
    def Export_csv(self):

        #Data
        if self.ui.radio_Data.isChecked():
            if self.ui.radio_DerNo.isChecked():
                df = self.Data
                Default_name = 'Data'
            elif self.ui.radio_Der1st.isChecked():
                df = self.Data_Der1st
                Default_name = 'Data_1stDer'
            elif self.ui.radio_Der2nd.isChecked():
                df = self.Data_Der2nd
                Default_name = 'Data_2ndDer'

        #Ratio
        if self.ui.radio_Rat.isChecked():
            if self.ui.radio_DerNo.isChecked():
                df = self.Ratio
                Default_name = 'Ratio'
            elif self.ui.radio_Der1st.isChecked():
                df = self.Ratio_Der1st
                Default_name = 'Ratio_1stDer'
            elif self.ui.radio_Der2nd.isChecked():
                df = self.Ratio_Der2nd
                Default_name = 'Ratio_2ndDer'

        #Ratio_AVR
        if self.ui.radio_AVR.isChecked():
            if self.ui.radio_DerNo.isChecked():
                df = self.Ratio_AVR
                Default_name = 'Ratio_AVR'
            elif self.ui.radio_Der1st.isChecked():
                df = self.Ratio_AVR_Der1st
                Default_name = 'Ratio_AVR_1stDer'
            elif self.ui.radio_Der2nd.isChecked():
                df = self.Ratio_AVR_Der2nd
                Default_name = 'Ratio_AVR_2ndDer'

        self.df = self.Ratio

        name = QFileDialog.getSaveFileName(self, 'Save File', filter='*.csv')
        if (name[0] == ''):
            pass
        else:
            if self.ui.checkBox_DataName.isChecked():
                df.to_csv(name[0][:-4] + '_' + Default_name + '.csv', sep='\t')
            else:
                df.to_csv(name[0],sep='\t')

###Example
            # if self.ui.radio_Rat.isChecked():  # Plot Ratio
            #     if self.ui.radio_DerNo.isChecked():
            # self.Data = self.meas.Field() / self.Data_Ref
            # self.Ratio = self.meas.Ratio() / self.Ratio_Ref
            # self.Ratio_AVR = self.Data.div(self.Data.mean(axis=1), axis=0)
            #
            # # Derivatives:
            # self.Data_Der1st = Treatment.derivative(self.Data)
            # self.Data_Der2nd = Treatment.derivative(self.Data_Der1st)
            #
            # self.Ratio_Der1st = Treatment.derivative(self.Ratio)
            # self.Ratio_Der2nd = Treatment.derivative(self.Ratio_Der1st)
            #
            # self.Ratio_AVR_Der1st = Treatment.derivative(self.Ratio_AVR)
            # self.Ratio_AVR_Der2nd = Treatment.derivative(self.Ratio_AVR_Der1st)


    ##############Start New Window#####################
    def Start_NWindow(self): #TODO work in progress
        # self.message("New detective has come to the town.")
        self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
        self.p.start("python3", ['MOD3p6.py'])

    def location_on_the_screen(self): #TODO make it work with PyQt6 !!!
        '''https://stackoverflow.com/questions/68037950/module-pyqt6-qtwidgets-has-no-attribute-qdesktopwidget'''
        ag = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        sg = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()

        # widget = self.geometry()
        # x = sg.width()/2 - widget.width()/2
        # y = sg.height()/2 - widget.height()/2
        # self.move(int(x), int(y))

    def centerWindow(self):
        window = CM()
        window.location_on_the_screen()


    ##############Load Stuff#####################
    def Load_Multiple(self, WL, names):
        self.unload_all(WL, names)

        Fnames = QFileDialog.getOpenFileNames()

        for i in range(0, np.size(Fnames[0])):
            WL.addItem(Fnames[0][i].split('/')[-1])

        WL.links = Fnames[0]
        # return Fnames

    # unload_all
    def unload_all(self, WidgetList, names):
        WL = WidgetList
        if WL.count() > 0:
            del names
            WL.clear()



# to run app
if __name__ == "__main__":
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # if hasattr(QtWidgets.QStyleFactory, "AA_UseHighDpiPixmaps"):
    #     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)  # instantiate a QtGui (holder for the app)
    window = CM()
    window.location_on_the_screen()
    window.show()
    sys.exit(app.exec())
