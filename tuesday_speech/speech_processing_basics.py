import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_filename = 'C:\\Users\\vvestman\\Desktop\\recordings\\9_jackson_0.wav'
target_sr = 8000

signal, sr = librosa.load(audio_filename, target_sr)


win_length = int(0.02 * sr)
hop_length = int(0.005 * sr)

spectrogram = np.absolute(librosa.stft(signal, n_fft=512, win_length=win_length, hop_length=hop_length, window='hamming'))

# EXERCISE: Write a code that creates a spectrogram without using librosa (framing --> windowing --> fft ...).

spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
librosa.display.specshow(spectrogram, y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

sparse_spectrogram = spectrogram[:, ::10]
plt.plot(np.linspace(0, 4000, spectrogram.shape[0]), sparse_spectrogram + np.arange(sparse_spectrogram.shape[1]) * 30)
plt.show()