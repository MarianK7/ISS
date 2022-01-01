from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import sys
import wave
import struct
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk


def DFT(x, N):
    coeffs = []
    e = np.exp(-1j * 2 * np.pi / N)
    for k in range(0, N // 2):
        coeffs.append(sum(x * [e ** (k * n) for n in range(N)]))
    return coeffs

###########################################################################################

fs, data = wavfile.read('audio/music.wav')

t = np.arange(data.size) / fs
tt = np.arange(0, 1024 / fs, 1 / fs)
samples = data.size
mi = data.min()
ma = data.max()

# Plot input U 4.1
#plt.figure(figsize=(8,4))
#plt.plot(t, data)
#plt.gca().set_xlabel('$t[s]$')
#plt.gca().set_ylabel('$Amplitúda$')
#plt.gca().set_title('Úloha 4.1')
#plt.savefig('u_4_1.pdf')
#plt.show()

###########################################################################################

print("==================================================")
print("Dĺžka nahrávky: ", t.max())
print("Počet vzorkov: ", data.size)
print("Vzorkovacia frekvencia: ", fs)
print("Minimum: ", mi)
print("Maximum: ", ma)

avg = sum(data) / fs
normData = data - avg
normData = normData / max(abs(normData))

# Plot normalized input
#plt.figure(figsize=(8,4))
#plt.plot(t, normData)
#plt.gca().set_title('Normalized data')
#plt.show()

###########################################################################################

frames = []
for i in range(samples // 512 - 1):
    frames.append(normData[i * 512: i * 512 + 1024])

print("Počet rámcov: ", len(frames))
print("Dĺžka rámca: ", len(frames[0]))

# Plot frame U 4.2
#plt.figure(figsize=(8,4))
#plt.plot(tt, frames[26])
#plt.gca().set_xlabel('$t[s]$')
#plt.gca().set_ylabel('$Amplitúda$')
#plt.gca().set_title('Úloha 4.2')
#plt.savefig('u_4_2.pdf')
#plt.show()

###########################################################################################

# FFT
maggsf = abs(np.array([np.fft.fft(frames[i])[0: 1024 // 2] for i in range(len(frames))]))

# DFT
coeffs = np.array([DFT(frames[i], 1024) for i in range(len(frames))])
maggs = abs(coeffs)
freqs = [k * fs // 1024 for k in range(1024 // 2)]

# Plot DFT U 4.3
#figure, ax = plt.subplots(2)
#ax[0].plot(freqs, maggs[26])
#ax[0].set_title('Úloha 4.3 DFT')
#ax[0].set_xlabel("Frekvencia[Hz]")
#ax[0].set_ylabel("Magnitúda")

#ax[1].plot(freqs, maggsf[26])
#ax[1].set_title('Úloha 4.3 FFT')
#ax[1].set_xlabel("Frekvencia[Hz]")
#ax[1].set_ylabel("Magnitúda")
#figure.tight_layout()
#plt.savefig('u_4_3.pdf')
#plt.show()

###########################################################################################

f, timeSpec, sgr = spectrogram(normData, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 

# Plot spectrogram U 4.4
#plt.figure(figsize=(10,5))
#plt.pcolormesh(timeSpec,f,sgr_log, shading='gouraud')
#plt.gca().set_xlabel('Čas [s]')
#plt.gca().set_ylabel('Frekvence [Hz]')
#plt.title("Úloha 4.4 spectrogram")
#cbar = plt.colorbar()
#cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
#plt.savefig('u_4_4.pdf')
#plt.show()

###########################################################################################

freks = [930,1860,2790,3720]
badFreqs = np.array(freks)
print("Špiny: ", badFreqs)

arr = []
for i in range(data.size):
  arr.append(i/fs)

outCos1 = np.cos(2 * np.pi * freks[0] * np.array(arr))
outCos2 = np.cos(2 * np.pi * freks[1] * np.array(arr))
outCos3 = np.cos(2 * np.pi * freks[2] * np.array(arr))
outCos4 = np.cos(2 * np.pi * freks[3] * np.array(arr))

cosines = outCos1 + outCos2 + outCos3 + outCos4

# U 4.6 generating disturbive cosines
wavfile.write('audio/cosin.wav', fs, cosines.astype(np.float32))

s, fs = sf.read('audio/cosin.wav')
t = np.arange(s.size) / fs

# Plot spectrogram of the cosines
f, timeSpec, sgr = spectrogram(s, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 

#plt.figure(figsize=(10,5))
#plt.pcolormesh(timeSpec,f,sgr_log, shading='gouraud')
#plt.gca().set_xlabel('Čas [s]')
#plt.gca().set_ylabel('Frekvence [Hz]')
#plt.title("Úloha 4.6 spectrogram")
#cbar = plt.colorbar()
#cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
#plt.savefig('u_4_6.pdf')
#plt.show()


###########################################################################################

print("Filters: ")
highPass = (badFreqs + 50) 
lowPass = (badFreqs - 50)
highBound = (badFreqs + 30)
lowBound = (badFreqs - 30)
zeros, poles, gains, bOut, aOut = [], [], [], [], []
for i in range(4):
    wp = [lowPass[i], highPass[i]]
    ws = [lowBound[i], highBound[i]]
    order = signal.buttord(wp, ws, 3, 40, analog = False, fs = fs)
    print("daco: ", order)
    z, p, k = signal.butter(order[0], order[1], btype = 'bandstop', analog = False, output = "zpk", fs = fs)
    b, a = signal.butter(order[0], order[1], btype = 'bandstop', analog = False, output = "ba", fs = fs)
    zeros.append(z)
    poles.append(p)
    gains.append(k)
    bOut.append(b)
    aOut.append(a)

# Plot zeros and poles U 4.8
#plt.figure(figsize=(6,6))
#ang = np.linspace(0, 2*np.pi,100)
#plt.plot(np.cos(ang), np.sin(ang))
#plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='r', label='nuly')
#plt.scatter(np.real(poles), np.imag(poles), marker='x', color='g', label='póly')
#plt.title("Úloha 4.8")
#plt.gca().set_xlabel('Realná složka R{Z}')
#plt.gca().set_ylabel('Imaginarní složka I{Z}')
#plt.grid(alpha=0.5, linestyle='--')
#plt.legend(loc='upper right')
#plt.savefig('u_4_8.pdf')
#plt.show()

sos = [signal.zpk2sos(zeros[i], poles[i], gains[i]) for i in range(4)]

# Print coeffs of the each filter U 4.7
#original_stdout = sys.stdout 
#with open('Coefficients.txt', 'w') as f:
#    sys.stdout = f 
#    print("#####################################################################")
#    print("Koeficienty prvého filtra:")
#    print(sos[0])
#    print("#####################################################################")
#    print("Koeficienty druhého filtra: ")
#    print(sos[0])
#    print("#####################################################################")
#    print("Koeficienty tretieho filtra: ")
#    print(sos[0])
#    print("#####################################################################")
#    print("Koeficienty štvrtého filtra: ")
#    print(sos[0])
#    print("#####################################################################")
#    sys.stdout = original_stdout

# Plot impulse response
timeResponse = np.linspace(0, 1, 700, False)
x = signal.unit_impulse(700)
#figure, ax = plt.subplots(4)

#ax[0].set_title("Úloha 4.7 filter č.1 (" + str(freks[0]) + " Hz)")
#ax[0].set_xlabel('Time[s]')
#ax[0].plot(timeResponse, signal.sosfilt(sos[0], x))

#ax[1].set_title("Úloha 4.7 filter č.2 (" + str(freks[1]) + " Hz)")
#ax[1].set_xlabel('Time[s]')
#ax[1].plot(timeResponse, signal.sosfilt(sos[1], x))

#ax[2].set_title("Úloha 4.7 filter č.3 (" + str(freks[2]) + " Hz)")
#ax[2].set_xlabel('Time[s]')
#ax[2].plot(timeResponse, signal.sosfilt(sos[2], x))

#ax[3].set_title("Úloha 4.7 filter č.4 (" + str(freks[3]) + " Hz)")
#ax[3].set_xlabel('Time[s]')
#ax[3].plot(timeResponse, signal.sosfilt(sos[3], x))

#plt.savefig('u_4_7.pdf')
#figure.tight_layout()
#plt.show()


# Plot frequency response U 4.9
#figure, ax = plt.subplots(4)

#w, h = signal.sosfreqz(sos[0], data.size, whole = False, fs = fs)
#ax[0].set_title("Úloha 4.9 filter č.1 (" + str(freks[0]) + " Hz)")
#ax[0].plot(w, abs(h))

#w, h = signal.sosfreqz(sos[1], data.size, whole = False, fs = fs)
#ax[1].set_title("Úloha 4.9 filter č.2 (" + str(freks[1]) + " Hz)")
#ax[1].plot(w, abs(h))

#w, h = signal.sosfreqz(sos[2], data.size, whole = False, fs = fs)
#ax[2].set_title("Úloha 4.9 filter č.3 (" + str(freks[2]) + " Hz)")
#ax[2].plot(w, abs(h))

#w, h = signal.sosfreqz(sos[3], data.size, whole = False, fs = fs)
#ax[3].set_title("Úloha 4.9 filter č.4 (" + str(freks[3]) + " Hz)")
#ax[3].plot(w, abs(h))

#plt.savefig('u_4_9.pdf')
#figure.tight_layout()
#plt.show()

# Filter the input data U 4.10
outData = normData
sf = normData
print("Delka (b): ", len(bOut), "Delka (a): ", len(aOut))

#print("filter (b): ", bOut, "filter (a): ", aOut)
for i in range(len(bOut)):
    sf = signal.lfilter(bOut[i], aOut[i], sf)

for i in sos:
    outData = signal.sosfilt(i, outData)

print("Velkost: ", outData.size)

wavfile.write('audio/out_test.wav', fs, outData.astype(np.float32))
