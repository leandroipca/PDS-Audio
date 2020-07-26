from collections import deque

import librosa
import pyaudio
import sounddevice as sd
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048
RECORD_SECONDS = 2.4
ativo = True

# d = deque()
#
# for x in range(20):
#     d.append(x)
#
# for y in range(4):
#     d.popleft()

def stream ():
    p = pyaudio.PyAudio()
    frames = deque()
    sound = []

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    data = stream.read(CHUNK)
    # data = sd.rec(samplerate=RATE, channels=1)
    frames.append(data)
    while ativo:
        if len(frames) > int(RATE * RECORD_SECONDS):
            sound = [frames for idx in range(int(RATE * RECORD_SECONDS))]
            plot(sound)
            print("* processar")

            for y in range(CHUNK*10):
                frames.popleft()
        else:
            sd.wait(CHUNK)

    return 0

    # print("* done recording")
    # # stream.stop_stream()
    # # stream.close()
    # # p.terminate()
    # print("A detetar!")
    # frames = sd.RawStream(device=None, samplerate=RATE, channels=1, blocksize=int(RATE/CHUNK * RECORD_SECONDS))
    #
    # frames = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=2)
    # sd.wait()
    # frames = librosa.to_mono(frames)
    # print("A processar!")

def plot(sound):
    plt.plot(sound)
    plt.xlabel('tempo(s)'), plt.ylabel('amplitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(sd.query_devices())
    stream()

