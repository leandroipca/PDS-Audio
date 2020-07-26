import pickle
import librosa
import pyaudio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import time
import wave
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048
RECORD_SECONDS = 2.3

scaler = StandardScaler()
scaler_filename = "treinoAudio/modelo_scaler_pessoa.bin"
scaler = pickle.load(open(scaler_filename,"rb"))
scaler_cmd = StandardScaler()
scaler_cmd_filename = "treinoAudio/modelo_scaler_comand.bin"
scaler_cmd = pickle.load(open(scaler_cmd_filename,"rb"))

mlp = MLPClassifier()
filename = "treinoAudio/treino_modelo_pessoa.bin"
mlp = pickle.load(open(filename,"rb"))
mlp_cmd = MLPClassifier()
filename_cmd = "treinoAudio/treino_modelo_comando.bin"
mlp_cmd = pickle.load(open(filename_cmd,"rb"))

grupo = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "Desc")
command = ("parar", "recuar", "direita", "esquerda", "baixo", "centro", "cima", "avançar", "Desc")

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE
        self.CHUNK = CHUNK
        self.p = None
        self.stream = None
        self.numpy_array = []
        self.numpy_arr = []
        # self.frames = []
        self.result = 8
        self.result_cmd = 8
        self.framesSize = 0

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)


    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        self.numpy_array.append(in_data)
        return None, pyaudio.paContinue

    def mainloop(self):

        self.npSize = len(self.numpy_array)

        if self.npSize >= RATE*RECORD_SECONDS/self.CHUNK:

            self.frames = self.numpy_array[int(self.npSize-(RATE*RECORD_SECONDS/(self.CHUNK))):]

            # Save the recorded data as a WAV file
            wf = wave.open("temp.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            # Leitura do ficheiro para teste
            self.data, self.samplerate = sf.read("temp.wav")
            self.X_person = prep_data(self.data)
            self.X_cmd = prep_data_cmd(self.data)
            self.result, self.result_cmd = prev_result(self.X_person, self.X_cmd)

            #Grafico
            plt.figure(figsize=(5, 1), dpi=50)
            XX = np.arange(0, len(self.data), 1)
            plt.plot(XX , self.data)
            plt.savefig("saida.png")
            plt.close()

        else:
            time.sleep(0.2)

        return self.result, self.result_cmd


def prep_data (clip_audio):
    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T,axis=0)
    X = []
    X.append(mfcc)
    X_person = scaler.transform(X)
    return X_person


def prep_data_cmd(clip_audio):
    W = []
    result = np.array([])

    mfcc = np.mean(librosa.feature.mfcc(clip_audio, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512).T, axis=0)
    result = np.hstack((result, mfcc))
    # stft = np.abs(librosa.stft(clip_audio))
    # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
    # result = np.hstack((result, chroma))
    # mel = np.mean(librosa.feature.melspectrogram(clip_audio, sr=RATE).T, axis=0)
    # result = np.hstack((result, mel))
    W.append(result)

    X_cmd = scaler_cmd.transform(W)
    return X_cmd


def prev_result (X_person, X_cmd):

    predictions = mlp.predict(X_person)
    a = predictions.item(0)

    confianca = mlp.predict_proba(X_cmd)
    if (confianca.item(a) > 0.01):
        resultado = grupo[a]
    else:
        resultado = " "

    print(resultado)
    print(confianca.item(0))

    predictions_cmd = mlp_cmd.predict(X_cmd)
    b = predictions_cmd.item(0)

    confiancacmd = mlp.predict_proba(X_cmd)
    if (confiancacmd.item(b)>0.01):
        resultadoCmd = command[b]
    else:
        resultadoCmd = " "

    print(resultadoCmd)
    print(confiancacmd.item(1))

    return resultado, resultadoCmd

