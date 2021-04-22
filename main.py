import sys, os, shutil
from librosa.core.spectrum import _spectrogram
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
    hours, remainder = divmod(ms, 3_600_000)
    minutes, r = divmod(remainder, 60_000)
    seconds, _ = divmod(remainder, 1_000)
    return ("%d:%02d:%02d" % (hours, minutes, seconds)) if hours else ("%d:%02d" % (minutes, seconds))

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

        
        self.samples = None
        self.sampling_rate = None
        self.samples_after = None

        self.first_turn = True              # This Prevents the Spectrogram range variables from being overwritten

        self.PLOT_DIR = 'Plots'
        self.PDF_DIR = 'PDFs'
        self.ui.save_session_data.clicked.connect(lambda: self.save_session())
        self.ui.actionSave.triggered.connect(lambda: self.save_session())

        self.audio_player_before = QMediaPlayer()
        self.audio_player_after = QMediaPlayer()
        self.audio_player_before.setNotifyInterval(1)
        self.audio_player_after.setNotifyInterval(1)

        self.bands_powers = [0.0, 0.25, 0.50, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]

        self.spectrogram_power_range = {'min': np.array([]), 'max': np.array([])}
        

        self.ui.min_pixel_intensity.valueChanged.connect(lambda: self.spectrogram_pixels_intensity('min'))
        self.ui.max_pixel_intensity.valueChanged.connect(lambda: self.spectrogram_pixels_intensity('max'))

        self.spectrogram_time_min, self.spectrogram_time_max = 0, 0             # Sync With Play

        self.band_slider = {}
        self.band_label = {}

        for index in range(10):
            self.band_slider[index] = getattr(self.ui, 'band_{}'.format(index+1))
            self.band_label[index] = getattr(self.ui, 'band_{}_label'.format(index+1))
            

        for slider in self.band_slider.values():
            slider.setDisabled(True)
            slider.setStyleSheet('selection-background-color: grey')

        for index, slider in self.band_slider.items():
            slider.valueChanged.connect(lambda _, index=index: self.slider_gain_updated(index))

        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]

        self.modified_signal = np.array([])
        self.current_slider_gain = [1.0] * 10

        self.controlers = {'before': [], 'after': []}

        for button, function in zip(['zoom_in', 'zoom_out', 'jump_forward', 'jump_back'], [self.zoomin, self.zoomout, self.forward, self.back]):
            self.controlers['before'].append((getattr(self.ui, '{}_btn_before'.format(button)), function))
            self.controlers['after'].append((getattr(self.ui, '{}_btn_after'.format(button)), function))

        for channel in self.controlers.values():
            for signal in channel:
                signal[0].clicked.connect(signal[1])

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

        self.playback_position = {
            'before': None,
            'after': None
        }

        self.time_seeker = {
            'before': self.ui.time_seeker_before,
            'after': self.ui.time_seeker_after
        }

        self.total_time = {
            'before': self.ui.total_time_before,
            'after': self.ui.total_time_after
        }

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionNew.triggered.connect(self.new_instance)
        self.ui.actionOpen.triggered.connect(self.open_audio_file)


        self.ui.play_btn_before.clicked.connect(self.audio_player_before.play)
        self.ui.pause_btn_before.clicked.connect(self.audio_player_before.pause)
        self.ui.stop_btn_before.clicked.connect(self.audio_player_before.stop)

        self.ui.play_btn_after.clicked.connect(self.audio_player_after.play)
        self.ui.pause_btn_after.clicked.connect(self.audio_player_after.pause)
        self.ui.stop_btn_after.clicked.connect(self.audio_player_after.stop)

        self.ui.palettes_box.currentTextChanged.connect(self.change_palette)

        self.ui.playback_speed_before.currentIndexChanged.connect(lambda: self.audio_player_before.setPlaybackRate(float(self.ui.playback_speed_before.currentText()[1:])))
        self.audio_player_before.durationChanged.connect(lambda duration: self.update_duration(duration, 'before'))

        self.ui.playback_speed_after.currentIndexChanged.connect(lambda: self.audio_player_after.setPlaybackRate(float(self.ui.playback_speed_after.currentText()[1:])))
        self.audio_player_after.durationChanged.connect(lambda duration: self.update_duration(duration, 'after'))


    def new_instance(self):
        self.new_instance = MainWindow()
        self.new_instance.show()

    def open_audio_file(self):
        path = qtw.QFileDialog.getOpenFileName(None, 'Load Audio', './', "Audio File(*.wav)")[0]

        for slider in self.band_slider.values():
            slider.setDisabled(False)
            slider.setStyleSheet('selection-background-color: blue')

        self.ui.max_pixel_intensity.setDisabled(False)
        self.ui.max_pixel_intensity.setStyleSheet('selection-background-color: blue')
        self.ui.min_pixel_intensity.setDisabled(False)
        self.ui.min_pixel_intensity.setStyleSheet('selection-background-color: blue')
        
        self.audio_player_before.setMedia(QMediaContent(qtc.QUrl.fromLocalFile(path)))
        self.audio_player_before.positionChanged.connect(lambda position: self.update_timestamp(position, self.ui.current_time_before, self.ui.time_seeker_before, 'before'))
        self.ui.time_seeker_before.valueChanged.connect(self.audio_player_before.setPosition)
        self.sampling_rate, self.samples = scipy.io.wavfile.read(path)
        self.plot_graph(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.modify_signal()

    def plot_graph(self, samples, sampling_rate, widget):
        peak_value = np.amax(samples)
        normalized_data = samples / peak_value
        length = samples.shape[0] / sampling_rate
        time = list(np.linspace(0, length, samples.shape[0]))

        drawing_pen = pg.mkPen(color=(255, 0, 0), width=0.5)
        self.plot_widget[widget].removeItem(self.data_line[widget])
        self.data_line[widget] = self.plot_widget[widget].plot(
            time, normalized_data, pen=drawing_pen)
        self.plot_widget[widget].plotItem.setLabel(axis='left', text='Normalized Amplitude')
        self.plot_widget[widget].plotItem.setLabel(axis='bottom', text='time [s]')
        self.plot_widget[widget].plotItem.getViewBox().setLimits(xMin=0, xMax=np.max(time), yMin=-1.1, yMax=1.1)
        self.spectrogram_time_min, self.spectrogram_time_max = self.plot_widget[widget].plotItem.getAxis('bottom').range
        self.playback_position[widget] = pyqtgraph.LinearRegionItem(values=(0, 0))
        self.plot_widget[widget].plotItem.getViewBox().addItem(self.playback_position[widget])


    def plot_spectrogram(self, samples, sampling_rate, widget):
        self.spectrogram_widget[widget].getFigure().clear()
        spectrogram_axes = self.spectrogram_widget[widget].getFigure().add_subplot(111)

        data = samples.astype('float32')
        frequency_magnitude = np.abs(librosa.stft(data))**2

        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_mels=128)

        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        if self.first_turn:
            min_intensity = np.ceil(np.amin(decibel_spectrogram))
            max_intensity = np.ceil(np.amax(decibel_spectrogram))

            self.spectrogram_power_range['min'] = np.linspace(min_intensity, min_intensity/2, 10).astype('int')
            self.spectrogram_power_range['min'] = np.append(self.spectrogram_power_range['min'], np.array([self.spectrogram_power_range['min'][0]]))
            self.spectrogram_power_range['max'] = np.linspace((min_intensity+1)/2, max_intensity, 10).astype('int')
            self.spectrogram_power_range['max'] = np.append(self.spectrogram_power_range['max'], np.array([self.spectrogram_power_range['max'][-1]]))
            self.ui.min_pixel_intensity_lab.setText(str(self.spectrogram_power_range['min'][-1]))
            self.ui.max_pixel_intensity_lab.setText(str(self.spectrogram_power_range['max'][-1]))
            self.first_turn = False

        spectrogram_image = librosa.display.specshow(decibel_spectrogram, x_axis='time', y_axis='mel', sr=sampling_rate,
                ax=spectrogram_axes, cmap=self.current_color_palette, vmin=self.spectrogram_power_range['min'][-1], vmax=self.spectrogram_power_range['max'][-1])

        self.spectrogram_widget[widget].getFigure().colorbar(spectrogram_image, ax=spectrogram_axes, format='%+2.0f dB')
        spectrogram_axes.set(xlim=[self.spectrogram_time_min, self.spectrogram_time_max])
        self.spectrogram_widget[widget].draw()

    def modify_signal(self):
        frequency_content = np.fft.rfftfreq(len(self.samples), d=1/self.sampling_rate)
        modified_signal = np.fft.rfft(self.samples)
        for index, slider_gain in enumerate(self.current_slider_gain):
            frequency_range_min = (index + 0) * self.sampling_rate / (2 * 10)
            frequency_range_max = (index + 1) * self.sampling_rate / (2 * 10)

            range_min_frequency = frequency_content > frequency_range_min
            range_max_frequency = frequency_content <= frequency_range_max
            
            slider_min_max = []
            for is_in_min_frequency, is_in_max_frequency in zip(range_min_frequency, range_max_frequency):
                slider_min_max.append(is_in_min_frequency and is_in_max_frequency)

            modified_signal[slider_min_max] *= slider_gain

        self.samples_after = np.fft.irfft(modified_signal)

        self.save_output_wav()

        self.plot_graph(self.samples_after, self.sampling_rate, 'after')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')

    def spectrogram_pixels_intensity(self, widget):
        slider = getattr(self.ui, '{}_pixel_intensity'.format(widget))
        self.spectrogram_power_range[widget][-1] = self.spectrogram_power_range[widget][int(slider.value())]
        label = getattr(self.ui, '{}_pixel_intensity_lab'.format(widget))
        label.setText(str(self.spectrogram_power_range[widget][-1]))
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
      
    def update_duration(self, duration, widget):
        self.time_seeker[widget].setMaximum(duration)

        if duration >= 0:
            self.total_time[widget].setText(time_stamp(duration))

    def update_timestamp(self, position, currentTimeLabel, timeSlider, widget):
        if position >= 0:
            currentTimeLabel.setText(time_stamp(position))

        timeSlider.blockSignals(True)
        timeSlider.setValue(position)
        timeSlider.blockSignals(False)

        self.playback_position[widget].setRegion((position/1000, position/1000))
        minRange, maxRange = self.plot_widget[widget].plotItem.getAxis('bottom').range
        if(position >= maxRange * 1000):
            self.plot_widget[widget].plotItem.getViewBox().translateBy((maxRange-minRange), 0)
            self.synchronize()

        if(position <= minRange * 1000):
            self.plot_widget[widget].plotItem.getViewBox().translateBy(-minRange)
            self.synchronize()

    hours

    def zoomin(self) -> None:
        self.ui.graph_before.plotItem.getViewBox().scaleBy((0.75, 1.0))
        self.synchronize()

    def zoomout(self) -> None:
        self.ui.graph_before.plotItem.getViewBox().scaleBy((1.25, 1.0))
        self.synchronize()

    def back(self):
        self.ui.graph_before.plotItem.getViewBox().translateBy((-0.5, 0.0))
        self.synchronize()

    def forward(self):
        self.ui.graph_before.plotItem.getViewBox().translateBy((0.5, 0.0))
        self.synchronize()

    def synchronize(self):
        self.ui.graph_before.plotItem.getViewBox().setXLink(self.ui.graph_after.plotItem)
        self.spectrogram_time_min, self.spectrogram_time_max = self.ui.graph_before.plotItem.getAxis('bottom').range
        self.plot_spectrogram(self.samples, self.sampling_rate, 'before')
        self.plot_spectrogram(self.samples_after, self.sampling_rate, 'after')


    def save_output_wav(self):
        try:
            shutil.rmtree('wav')
            os.mkdir('wav')
        except:
            os.mkdir('wav')
        self.now = datetime.now()
        self.now = f'{self.now:%Y-%m-%d %H-%M-%S.%f %p}'
        scipy.io.wavfile.write(f"wav/SBME{self.now}.wav", self.sampling_rate, self.samples_after.astype(np.int16))
        path = os.listdir('wav')
        self.audio_player_after.setMedia(QMediaContent(qtc.QUrl.fromLocalFile(f'wav/{path[0]}')))
        self.audio_player_after.positionChanged.connect(lambda position: self.update_timestamp(position, self.ui.current_time_after, self.ui.time_seeker_after, 'after'))
        self.ui.time_seeker_after.valueChanged.connect(self.audio_player_after.setPosition)

    def save_session(self):
        if not self.sampling_rate:
            qtw.QMessageBox.information(
                self, 'failed', 'You have to plot a signal first')
            return

        try:
            shutil.rmtree(self.PLOT_DIR)
            os.mkdir(self.PLOT_DIR)
        except FileNotFoundError:
            os.mkdir(self.PLOT_DIR)

        for index, channel in enumerate(['before', 'after']):
            exporter = pg.exporters.ImageExporter(
                self.plot_widget[channel].scene())
            exporter.export(f'{self.PLOT_DIR}/plot-{index}.png')

            self.spectrogram_widget[channel].fig.savefig(f'{self.PLOT_DIR}/spec-{index}.png')

        pdf = PDF()
        plots_per_page = pdf.construct(self.PLOT_DIR)

        for page_images in plots_per_page:
            pdf.print_page(page_images, self.PLOT_DIR)

        outFile = qtw.QFileDialog.getSaveFileName(
            None, 'Save Session', './', "PDF File(*.pdf)")
        pdf.output(outFile[0], 'F')
        try:
            shutil.rmtree(self.PLOT_DIR)
        except:
            pass

        qtw.QMessageBox.information(self, 'success', 'PDF has been created')

        sampling_rate, samples = scipy.io.wavfile.read(f"wav/SBME{self.now}.wav")
        outFile = qtw.QFileDialog.getSaveFileName(
            None, 'Save Session', './', "Wav File(*.wav)")
        scipy.io.wavfile.write(outFile[0], sampling_rate, samples.astype(np.int16))
        qtw.QMessageBox.information(self, 'success', 'Wav has been saved')
        
        

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    app.setStyle("Fusion")
    mw = MainWindow()
    sys.exit(app.exec_())