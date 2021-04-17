import sys, os, shutil
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from datetime import datetime
import pyqtgraph
import pyqtgraph.exporters
import matplotlib.pyplot as plt
from main_layout import Ui_MainWindow
import scipy.io.wavfile
import librosa.display
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PDF import PDF

def time_stamp(ms):
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, _ = divmod(r, 1_000)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

        self.PLOT_DIR = 'Plots'
        self.PDF_DIR = 'PDFs'
        self.amplitude = [[], []]
        self.graphChannels = [self.ui.graph_before,self.ui.graph_after]
        self.spectrogramChannels = [self.ui.spectrogram_before, self.ui.spectrogram_after]
        self.ui.pushButton.clicked.connect(lambda: self.generatePDF())
        self.ui.actionSave.triggered.connect(lambda: self.generatePDF())

        
        self.audio_player_before = QMediaPlayer()
        self.audio_player_after = QMediaPlayer()
        self.plot = {'before': self.ui.graph_before, 'after': self.ui.graph_after}
        self.spectrogram = {'before': self.ui.spectrogram_before, 'after': self.ui.spectrogram_after}

        self.bands_powers = [0.0, 0.25, 0.50, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.bands = [None] * 10

        self.band_slider = {
            0: self.ui.band_1, 1: self.ui.band_2,
            2: self.ui.band_3, 3: self.ui.band_4,
            4: self.ui.band_5, 5: self.ui.band_6,
            6: self.ui.band_7, 7: self.ui.band_8,
            8: self.ui.band_9, 9: self.ui.band_10,
        }

        self.min_slider_ = [0, 2205, 4410, 6615, 8820, 11025, 13230, 15435, 17640, 19845, 22050] 
        self.max_silder_ = [0, 2205, 4410, 6615, 8820, 11025, 13230, 15435, 17640, 19845, 22050]
        self.current_min = 0
        self.current_max = 22050
        
        self.ui.max_freq_content.setDisabled(True)
        self.ui.max_freq_content.setStyleSheet('selection-background-color: grey')
        self.ui.min_freq_content.setDisabled(True)
        self.ui.min_freq_content.setStyleSheet('selection-background-color: grey')


        self.ui.max_freq_content.valueChanged.connect(lambda: self.range_max())
        self.ui.min_freq_content.valueChanged.connect(lambda: self.range_min())

        self.spectrogram_time_min, self.spectrogram_time_max = 0, 0
        for slider in self.band_slider.values():
            slider.setDisabled(True)
            slider.setStyleSheet('selection-background-color: grey');

        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]

        self.ui.band_1.valueChanged.connect(lambda: self.slider_gain_updated(0))
        self.ui.band_2.valueChanged.connect(lambda: self.slider_gain_updated(1))
        self.ui.band_3.valueChanged.connect(lambda: self.slider_gain_updated(2))
        self.ui.band_4.valueChanged.connect(lambda: self.slider_gain_updated(3))
        self.ui.band_5.valueChanged.connect(lambda: self.slider_gain_updated(4))
        self.ui.band_6.valueChanged.connect(lambda: self.slider_gain_updated(5))
        self.ui.band_7.valueChanged.connect(lambda: self.slider_gain_updated(6))
        self.ui.band_8.valueChanged.connect(lambda: self.slider_gain_updated(7))
        self.ui.band_9.valueChanged.connect(lambda: self.slider_gain_updated(8))
        self.ui.band_10.valueChanged.connect(lambda: self.slider_gain_updated(9))


        self.filtered_bands = []
        self.modified_signal = np.array([])
        self.current_slider_gain = [1.0] * 10
        

        self.band_label = {
            0: self.ui.band_1_label, 1: self.ui.band_2_label,
            2: self.ui.band_3_label, 3: self.ui.band_4_label,
            4: self.ui.band_5_label, 5: self.ui.band_6_label,
            6: self.ui.band_7_label, 7: self.ui.band_8_label,
            8: self.ui.band_9_label, 9: self.ui.band_10_label,
        }
        
        self.controls = {
            'before': {
                'play':self.ui.play_btn_before,
                'pause': self.ui.pause_btn_before, 
                'stop': self.ui.stop_btn_before,
                'jump_forward': self.ui.jump_forward_btn_before,
                'jump_backward': self.ui.jump_back_btn_before,
                'current_time': self.ui.current_time_before,
                'total_time': self.ui.total_time_before,
                'time_slider': self.ui.time_seeker_before
            },
            'after':{
                'play':self.ui.play_btn_after, 
                'pause': self.ui.pause_btn_after,
                'stop': self.ui.stop_btn_after,
                'jump_forward': self.ui.jump_forward_btn_after,
                'jump_backward': self.ui.jump_back_btn_after,
                'current_time': self.ui.current_time_after,
                'total_time': self.ui.total_time_after,
                'time_slider': self.ui.time_seeker_after
                }
            }

        self.plot_widget = {
            'before': self.ui.graph_before,
            'after': self.ui.graph_after
        }

        self.spectrogram_widget = {
            'before': self.ui.spectrogram_before,
            'after': self.ui.spectrogram_after
        }

        self.data_line = {
            'before': None,
            'after': None
        }

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionNew.triggered.connect(self.new_instance)
        self.ui.actionOpen.triggered.connect(self.open_audio_file)

        self.ui.zoom_in_btn_after.clicked.connect(self.zoomin)
        self.ui.zoom_in_btn_before.clicked.connect(self.zoomin)

        self.ui.zoom_out_btn_after.clicked.connect(self.zoomout)
        self.ui.zoom_out_btn_before.clicked.connect(self.zoomout)

        self.ui.jump_back_btn_before.clicked.connect(self.back)
        self.ui.jump_back_btn_after.clicked.connect(self.back)

        self.ui.jump_forward_btn_before.clicked.connect(self.forward)
        self.ui.jump_forward_btn_after.clicked.connect(self.forward)

        self.ui.play_btn_before.clicked.connect(self.audio_player_before.play)
        self.ui.pause_btn_before.clicked.connect(self.audio_player_before.pause)
        self.ui.stop_btn_before.clicked.connect(self.audio_player_before.stop)

        self.ui.play_btn_after.clicked.connect(self.audio_player_after.play)
        self.ui.pause_btn_after.clicked.connect(self.audio_player_after.pause)
        self.ui.stop_btn_after.clicked.connect(self.audio_player_after.stop)

        self.ui.palettes_box.currentTextChanged.connect(self.change_palette)

        self.ui.playback_speed_before.currentIndexChanged.connect(lambda: self.audio_player_before.setPlaybackRate(float(self.ui.playback_speed_before.currentText()[1:])))
        self.audio_player_before.durationChanged.connect(lambda duration: self.update_duration(duration))

        self.ui.playback_speed_after.currentIndexChanged.connect(lambda: self.audio_player_after.setPlaybackRate(float(self.ui.playback_speed_after.currentText()[1:])))
        self.audio_player_after.durationChanged.connect(lambda duration: self.update_duration_after(duration))

    def range_max(self):
        self.current_max = self.max_silder_[int(self.ui.max_freq_content.value())]
        self.ui.max_freq_content_lab.setText(str(self.current_max))
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def range_min(self):
        self.current_min = self.min_slider_[int(self.ui.min_freq_content.value())]
        self.ui.min_freq_content_lab.setText(str(self.current_min))
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')
        

    def change_palette(self):
        self.current_color_palette = self.available_palettes[self.ui.palettes_box.currentIndex()]
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def slider_gain_updated(self, index):
        slider_gain = self.bands_powers[self.band_slider[index].value()]
        self.band_label[index].setText(f'{slider_gain}')
        self.current_slider_gain[index] = slider_gain
        self.modify_signal()
        


    def modify_signal(self):
        frequency_content = np.fft.rfftfreq(len(self.samples), d=1/self.sampling_rate)
        modified_signal = np.fft.rfft(self.samples)
        for index, slider_gain in enumerate(self.current_slider_gain):
            frequency_range_min = (index + 0) * self.sampling_rate / (2 * 10)
            frequency_range_max = (index + 1) * self.sampling_rate / (2 * 10)

            x = frequency_content > frequency_range_min
            y = frequency_content <= frequency_range_max
            
            z = []
            for a, b in zip(x, y):
                z.append(a and b)


            modified_signal[z] *= slider_gain

        self.samples_after = np.fft.irfft(modified_signal)

        samplerate = 44100
        try:
            shutil.rmtree('wav')
            os.mkdir('wav')
        except:
            os.mkdir('wav')
        now = datetime.now()
        now = f'{now:%Y-%m-%d %H-%M-%S.%f %p}'
        scipy.io.wavfile.write(f"wav/SBME{now}.wav", samplerate, self.samples_after.astype(np.int16))
        path = os.listdir('wav')
        self.audio_player_after.setMedia(QMediaContent(qtc.QUrl.fromLocalFile(f'wav/{path[0]}')))
        self.audio_player_after.positionChanged.connect(lambda position: self.update_timestamp(position, self.ui.current_time_after, self.ui.time_seeker_after))
        self.ui.time_seeker_after.valueChanged.connect(self.audio_player_after.setPosition)

        self.plot_graph(self.samples_after, self.sampling_rate, 'after')
        

    
    def open_audio_file(self):
        filename = qtw.QFileDialog.getOpenFileName(
            None, 'Load Audio', './', "Audio File(*.wav)")
        path = filename[0]

        
        self.audio_player_before.setMedia(QMediaContent(qtc.QUrl.fromLocalFile(path)))
        self.sampling_rate, self.samples = scipy.io.wavfile.read(path)
        self.plot_graph(self.samples, self.sampling_rate, 'before')
        self.audio_player_before.positionChanged.connect(lambda position: self.update_timestamp(position, self.ui.current_time_before, self.ui.time_seeker_before))
        self.ui.time_seeker_before.valueChanged.connect(self.audio_player_before.setPosition)
        for slider in self.band_slider.values():
            slider.setDisabled(False)
            slider.setStyleSheet('selection-background-color: blue')
        self.ui.max_freq_content.setDisabled(False)
        self.ui.max_freq_content.setStyleSheet('selection-background-color: blue')
        self.ui.min_freq_content.setDisabled(False)
        self.ui.min_freq_content.setStyleSheet('selection-background-color: blue')
        self.modify_signal()
      
    def update_duration(self, duration):
        self.ui.time_seeker_before.setMaximum(duration)

        if duration >= 0:
            self.ui.total_time_before.setText(time_stamp(duration))
    
    def update_duration_after(self, duration):
        self.ui.time_seeker_after.setMaximum(duration)

        if duration >= 0:
            self.ui.total_time_after.setText(time_stamp(duration))

    def update_timestamp(self, position, currentTimeLabel, timeSlider):
        if position >= 0:
            currentTimeLabel.setText(time_stamp(position))

        timeSlider.blockSignals(True)
        timeSlider.setValue(position)
        timeSlider.blockSignals(False)

    def plot_graph(self, samples, sampling_rate, widget):
        # Preparing received data
        peak_value = np.amax(samples)
        normalized_data = samples / peak_value
        length = samples.shape[0] / sampling_rate
        time = list(np.linspace(0, length, normalized_data.shape[0]))

        drawing_pen = pg.mkPen(color=(255, 0, 0), width=0.5)
        self.plot_widget[widget].removeItem(self.data_line[widget])
        self.data_line[widget] = self.plot_widget[widget].plot(
            time, normalized_data, pen=drawing_pen)
        self.plot_widget[widget].plotItem.setLabel(axis='left', text='Normalized Amplitude')
        self.plot_widget[widget].plotItem.setLabel(axis='bottom', text='time [s]')
        self.plot_widget[widget].plotItem.getViewBox().setLimits(xMin=0, xMax=np.max(time), yMin=-1.1, yMax=1.1)
        self.spectrogram_time_min, self.spectrogram_time_max = self.plot_widget[widget].plotItem.getAxis('bottom').range
        # self.playback_position[widget] = pyqtgraph.LinearRegionItem(values=(0, 0))
        # self.plot_widget[widget].plotItem.getViewBox().addItem(self.playback_position[widget])
            

        self.plot_spectrogram(samples, sampling_rate, widget)

    def plot_spectrogram(self, samples, sampling_rate, widget):
        axes = self.spectrogram_widget[widget].getFigure().clear()
        axes = self.spectrogram_widget[widget].getFigure().add_subplot(111)

        data = samples.astype('float32')
        D = np.abs(librosa.stft(data))**2

        S = librosa.feature.melspectrogram(S=D, y=data, sr=sampling_rate, n_mels=128,
                                            fmin=self.current_min ,fmax=self.current_max)

        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, x_axis='time',

                         y_axis='mel', sr=sampling_rate,

                        fmin=self.current_min ,fmax=self.current_max,  ax=axes, cmap=self.current_color_palette)
                         
        self.spectrogram_widget[widget].getFigure().colorbar(img, ax=axes, format='%+2.0f dB')
        axes.set(xlim=[self.spectrogram_time_min, self.spectrogram_time_max])
        self.spectrogram_widget[widget].draw()

    def zoomin(self) -> None:
        self.ui.graph_before.plotItem.getViewBox().scaleBy((0.75, 1.0))
        self.ui.graph_before.plotItem.getViewBox().setXLink(self.ui.graph_after.plotItem)
        self.spectrogram_time_min, self.spectrogram_time_max = self.ui.graph_before.plotItem.getAxis('bottom').range
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def zoomout(self) -> None:
        self.ui.graph_before.plotItem.getViewBox().scaleBy((1.25, 1.0))
        self.ui.graph_before.plotItem.getViewBox().setXLink(self.ui.graph_after.plotItem)
        self.spectrogram_time_min, self.spectrogram_time_max = self.ui.graph_before.plotItem.getAxis('bottom').range
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def back(self):
        self.ui.graph_before.plotItem.getViewBox().translateBy((-0.5, 0.0))
        self.ui.graph_before.plotItem.getViewBox().setXLink(self.ui.graph_after.plotItem)
        self.spectrogram_time_min, self.spectrogram_time_max = self.ui.graph_before.plotItem.getAxis('bottom').range
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def forward(self):
        self.ui.graph_before.plotItem.getViewBox().translateBy((0.5, 0.0))
        self.ui.graph_before.plotItem.getViewBox().setXLink(self.ui.graph_after.plotItem)
        self.spectrogram_time_min, self.spectrogram_time_max = self.ui.graph_before.plotItem.getAxis('bottom').range
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')


    def generatePDF(self):
        images = [0, 0]
        Idx = 0
        for channel in range(2):
            self.amplitude[channel]
            images[channel] = 1
            Idx += 1
            

        if not Idx:
            qtw.QMessageBox.information(
                self, 'failed', 'You have to plot a signal first')
            return

        try:
            shutil.rmtree(self.PLOT_DIR)
            os.mkdir(self.PLOT_DIR)
        except FileNotFoundError:
            os.mkdir(self.PLOT_DIR)

        for channel in range(2):
            if images[channel]:

                exporter = pg.exporters.ImageExporter(
                    self.graphChannels[channel].scene())
                exporter.export(f'{self.PLOT_DIR}/plot-{channel}.png')

                self.spectrogramChannels[channel].fig.savefig(f'{self.PLOT_DIR}/spec-{channel}.png')

        pdf = PDF()
        plots_per_page = pdf.construct(self.PLOT_DIR)
        [['plot1', 'spec1'], ['plot2', 'spec2']]

        for elem in plots_per_page:
            pdf.print_page(elem, self.PLOT_DIR)

        now = datetime.now()
        now = f'{now:%Y-%m-%d %H-%M-%S.%f %p}'
        try:
            os.mkdir(self.PDF_DIR)
        except:
            pass
        pdf.output(f'{self.PDF_DIR}/{now}.pdf', 'F')
        try:
            shutil.rmtree(self.PLOT_DIR)
        except:
            pass

        qtw.QMessageBox.information(self, 'success', 'PDF has been created')
  

    def new_instance(self):
        self.new_instance = MainWindow()
        self.new_instance.show()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    app.setStyle("Fusion")
    mw = MainWindow()
    sys.exit(app.exec_())