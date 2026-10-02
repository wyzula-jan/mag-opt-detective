from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    # def data(self, index, role):
    #     if role == Qt.DisplayRole:
    #         value = self._data.iloc[index.row(), index.column()]
    #         return str(value)
    #
    # def rowCount(self, index):
    #     return self._data.shape[0]
    #
    # def columnCount(self, index):
    #     return self._data.shape[1]
    #
    # def headerData(self, section, orientation, role):
    #     # section is the index of the column/row.
    #     if role == Qt.DisplayRole:
    #         if orientation == Qt.Horizontal:
    #             return str(self._data.columns[section])
    #
    #         if orientation == Qt.Vertical:
    #             return str(self._data.index[section])

class ListBoxWidget(QListWidget):
    #from https://learndataanalysis.org/implement-files-and-urls-to-listbox-widget-drag-and-drop-function-pyqt5-tutorial/
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.links = None
        self.resize(600, 600)

    def unload_all(self, event):
        if event.mimeData().hasUrls():
            self.clear()
            self.links = None
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction()
            # event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.unload_all(event)
        if event.mimeData().hasUrls():
            event.setDropAction()
            # event.setDropAction(Qt.CopyAction)
            event.accept()

            self.links = []
            self.links_view = []
            for url in event.mimeData().urls():
                # https://doc.qt.io/qt-5/qurl.html
                if url.isLocalFile():
                    self.links.append(str(url.toLocalFile()))
                    self.links_view.append((str(url.toLocalFile()).split('/')[-1]))
                    # print(str(url.toLocalFile()))
                else:
                    self.links.append(str(url.toString()))
                    self.links_view.append(str(url.toString()).split('/')[-1])
            self.addItems(self.links_view)
        else:
            event.ignore()
