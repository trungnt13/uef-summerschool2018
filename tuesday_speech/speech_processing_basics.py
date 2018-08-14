"https://github.com/Jakobovski/free-spoken-digit-dataset"

import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_filename = r'C:\Users\vvestman\Desktop\free-spoken-digit-dataset-master\recordings\9_jackson_0.wav'
target_sr = 8000

signal, sr = librosa.load(audio_filename, target_sr)


win_length = int(0.02 * sr)
hop_length = int(0.005 * sr)

spectrogram = librosa.stft(signal, n_fft=512, win_length=win_length, hop_length=hop_length, window='hamming')

# EXERCISE: Write a code that creates a spectrogram without using librosa (framing --> windowing --> fft ...).


frame_endpoints = list(range(win_length-1, signal.size, hop_length))
frames = np.zeros(shape=(win_length, len(frame_endpoints)))
for i in range(len(frame_endpoints)):
    frames[:, i] = signal[frame_endpoints[i] - win_length+1 : 
        frame_endpoints[i] + 1]

hamming_window = np.hamming(win_length)
frames = frames * hamming_window[:, None]      
spectrogram = np.fft.fft(frames, n=512, axis=0)
spectrogram = spectrogram[:257, :]

plt.figure()
plt.subplot(2,1,1)
plt.plot(signal)
plt.subplot(2,1,2)
plt.plot(frames[:, 40])

plt.figure()
spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
librosa.display.specshow(spectrogram, y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

sparse_spectrogram = spectrogram[:, ::10]
plt.plot(np.linspace(0, 4000, spectrogram.shape[0]), sparse_spectrogram + np.arange(sparse_spectrogram.shape[1]) * 30)
plt.show()
