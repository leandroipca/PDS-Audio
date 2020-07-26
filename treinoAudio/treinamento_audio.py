import numpy as np
import os
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from treinoAudio.AudioAugmentation import AudioAugmentation

DATASET_PATH = "../dataSetAudio/grupos"
SAMPLE_RATE = 44100
grupo = []
Naug = 1  # Maximo 11
Segundos = 2.3
aa = AudioAugmentation()


def procdados(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512):
    x = []
    y = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        semantic_label = dirpath.split("\\")[-1]
        grupo.append(semantic_label)
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            data = aa.read_audio_file(file_path, SAMPLE_RATE, Segundos)
            a = 0
            while True:
                # Adding noise to sound
                # data_noise = aa.add_noise(data, a)

                # Adding shift to sound
                data_shift = aa.shift(data, a)
                mfcc = np.mean(librosa.feature.mfcc(data_shift, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length).T, axis=0)
                x.append(mfcc)
                y.append(i - 1)

                # Adding stretch sound
                # data_stretch = aa.stretch(data)

                if a == Naug:
                    break
                else:
                    a += 1

            print("Numero de repeticoes: ", a)
            print("{}".format(file_path))

    #         if adicaoMin < adicaoControlo:
    #             adicaocontrolo = adicaoMin
    #
    # print("Adição minima: ", adicaoControlo)

    return x, y


if __name__ == "__main__":
    X, y = procdados(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    sc_filename = 'modelo_scaler_pessoa.bin'
    pickle.dump(scaler, open(sc_filename, 'wb'))

    mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size=30, beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(52, 26, 13), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=7,
                        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=False, warm_start=False)
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)

    filename = 'treino_modelo_pessoa.bin'
    pickle.dump(mlp, open(filename, 'wb'))

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(grupo)
