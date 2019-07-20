'''
Voronoi tiles, python metal implementation
added to runmetal

 def runThread(self, cbuffer, func, buffers, threads=None, label=None):
        desc = Metal.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()

        # threads

        # number of thread per group
        w = state.threadExecutionWidth()
        h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
        log.debug("w,h=%d,%d, bufmax=%d", w, h, bufmax)
        tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})

        # number of thread per grig
        ntg = self.getmtlsize(threads)
        log.debug("threads: %s %s", ntg, tpg)

        encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        log.debug("encode(compute) %s", label)
        encoder.endEncoding()
'''
from PyQt5.QtWidgets import QMainWindow, QLabel, QSizePolicy, QApplication, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import numpy as np
import sys
import runmetal


class Voronoi():
    def __init__(self, w, h):
        self.w, self.h = w, h

        n_points = min(w, h)
        points = np.empty(shape=(n_points, 2), dtype=np.int32)  # set of x,y points in w,h range
        points[:, 0] = np.random.randint(low=0, high=w, size=n_points, dtype=np.int32)
        points[:, 1] = np.random.randint(low=0, high=h, size=n_points, dtype=np.int32)

        self.pm, self.fn = None, None

        self.init_metal()

        self.pixelsBuf = self.pm.numpybuffer(np.empty(shape=(w, h), dtype=np.uint32))  # generate input MTL buffers
        colorBuf = self.pm.numpybuffer(np.random.randint(low=0, high=0xffffff, size=n_points, dtype=np.uint32))
        pointsBuf = self.pm.numpybuffer(points)
        countBuf = self.intBuffer(n_points)

        self.run_metal([self.pixelsBuf, pointsBuf, colorBuf, countBuf])

    def getPixels(self):
        return np.array(self.pm.buf2numpy(self.pixelsBuf, dtype=np.uint8).reshape(self.h, self.w, 4)[..., :3])

    def intBuffer(self, i):
        return self.pm.numpybuffer(np.array(i, dtype=np.int32))

    def init_metal(self):
        self.pm = runmetal.PyMetal()
        self.pm.opendevice()
        self.pm.openlibrary(filename='Voronoi.metal')
        self.fn = self.pm.getfn("Voronoi")

    def run_metal(self, parameters):
        pm = self.pm
        cqueue, cbuffer = pm.getqueue()

        pm.runThread(cbuffer=cbuffer, func=self.fn, buffers=parameters,
                     threads=({"width": self.w, "height": self.h, "depth": 1}))

        pm.enqueue_blit(cbuffer, self.pixelsBuf)

        pm.start_process(cbuffer)
        pm.wait_process(cbuffer)


class VoronoiWindow(QMainWindow):

    def __init__(self, w, h):
        super().__init__()
        self.w, self.h = w, h
        self.needsUpdate = True
        self.initUI()

    def initUI(self):
        self.setGeometry(10, 10, self.w, self.h)

        self.pxlbl = QLabel()

        self.setCentralWidget(self.pxlbl)
        self.center()
        self.show()

    def center(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def resizeEvent(self, event):
        self.setWindowTitle(f'Voronoi tiles - METAL ({self.w:} x {self.h:})')
        self.needsUpdate = True

    def paintEvent(self, event):
        def generate_pixmap():
            pixels = Voronoi(self.w, self.h).getPixels()
            qimage = QImage(pixels, pixels.shape[1], pixels.shape[0], QImage.Format_RGB888)
            return QPixmap(qimage).scaled(self.w, self.h, Qt.KeepAspectRatio)

        if self.needsUpdate:
            self.needsUpdate = False
            g = self.geometry()
            self.w, self.h = g.width(), g.height()

            self.pxlbl.setPixmap(generate_pixmap())


def main():
    app = QApplication(sys.argv)
    _ = VoronoiWindow(1024, 768)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
