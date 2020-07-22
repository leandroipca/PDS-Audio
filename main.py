# Script do trabalho prático de Laboratório Integrados II - Leandro Montenegro Pinto - 16207

# Imports das Bibliotecas
import sys
from PyQt5 import QtWidgets, uic, QtGui
import cv2
import os
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from Previsao.previsao_som import AudioHandler
import time
from PyQt5.QtGui import QPixmap, QImage


# Escolha do método de reconhecimento - LBPH
lbph = cv2.face.LBPHFaceRecognizer_create()

ah = AudioHandler()


def img3pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = channel * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


# Reconhecimento pela Webcam
def rec_lbph_window():

    if not camera.isOpened():
        camera.open(0)
        window.labelText.setText("Turning Camera ON")

    # Cores em BGR
    branco = [255, 255, 255]
    vermelho_escuro = [0, 0, 139]


    # Escolher qual metodo de detecção de faces
    detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
    # detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_alt.xml")
    # detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalcatface.xml")


    # A utilizar o LBPH com as características
    reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 100)


    # Escolher a base treinamento
    reconhecedor.read("classificadorLBPH.yml")

    # Escolher a escala de redimensionamento
    largura, altura = 220, 220

    # Escolha da fonte
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while (True):
        # Leitura da WebCam
        conectado, imagem = camera.read()

        #Converter em escala de cinzento
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Metodo de detecção de faces
        facesDetectadas = detectorFace.detectMultiScale(imagemCinza)
        for (x, y, l, a) in facesDetectadas:

            # A realizar o desenho da deteção da face
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x, y), (x + l, y + a), vermelho_escuro, 2)

            # Recolher dados após Predict
            id, confianca = reconhecedor.predict(imagemFace)

            # Verificação se o rosto detectado pertence ou não ao grupo selecionado
            if id == 8:
                nome = 'Grupo 8'
            else:
                nome = 'Nao Identificado'
            cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, branco)
            cv2.putText(imagem, 'Confianca: ', (x, y + (a + 50)), font, 1, branco)
            cv2.putText(imagem, str(round(confianca)), (x + 150, y + (a + 50)), font, 1, branco)

        cv2.imshow("Reconhecimento por LBPH", imagem)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            window.labelText.setText("Camera desligada")
            break
    camera.release()
    cv2.destroyAllWindows()

    return imagem

# Reconhecimento por imagens (carregadas)
def rec_lbph_imagens():

    # Cores em BGR
    branco = [255, 255, 255]
    vermelho_escuro = [0, 0, 139]


    # Metodo de detecção de faces
    detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")

    # A utilizar o LBPH com as características
    reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)

    # Escolher a base treinamento
    reconhecedor.read("classificadorLBPH.yml")

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fname = QFileDialog.getOpenFileName()

    while (True):

        # Leitura da imagem escolhida
        imagem = cv2.imread(fname[0])

        # Converter em escala de cinzento
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Metodo de detecção de faces
        facesDetectadas = detectorFace.detectMultiScale(imagemCinza)
        for (x, y, l, a) in facesDetectadas:

            # A realizar o desenho da deteção da face
            cv2.rectangle(imagem, (x, y), (x + l, y + a), vermelho_escuro, 2)

            # Recolher dados após Predict
            id, confianca = reconhecedor.predict(imagemCinza)

            # Verificação se o rosto detectado pertence ou não ao grupo selecionado
            nome = ""
            if id == 8:
                nome = 'Grupo 8'
            else:
                nome = 'Nao Identificado'
            cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, branco)
            cv2.putText(imagem, 'Confianca: ', (x, y + (a + 50)), font, 1, branco)
            cv2.putText(imagem, str(round(confianca)), (x + 150, y + (a + 50)), font, 1, branco)

        cv2.imshow("Reconhecimento por LBPH", imagem)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def rec_audio():

    # Inicia o stream de audio
    ah.start()

    while (True):

        # Previsão Som
        grupo, comando = ah.mainloop()
        if (type(grupo) == str) and grupo != " ":
            window.Pessoa.setText(grupo)
        if (type(comando) == str) and comando != " ":
            window.Comando.setText(comando)

        graf = cv2.imread("saida.png")
        window.graph1.setPixmap(img3pixmap(graf))

        time.sleep(0.2)

        if cv2.waitKey(1) == ord('q'):
            break

    """
    camera.release()
    cv2.destroyAllWindows()
    window.Video.setPixmap(img2pixmap(img))
    """

# Função do botão Reconhecer Web
def on_cameraON_clicked():
    window.labelText.setText("Reconhecimento em andamento")
    rec_lbph_window()

# Função do botão Reconhecer Imagem
def RecImagem_clicked():
    window.labelText.setText("Reconhecimento em andamento")
    rec_lbph_imagens()

# Função do botão sair
def sair_clicked():
    window.close()

# Função do botão Iniciar Audio
def start_clicked():
    rec_audio()

# Função do botão parar audio
def stop_clicked():
    window.close()
    ah.stop()
    sys.exit(0)

# Função do botão Treinamento
def treinamento():

    caminhos = [os.path.join('dataSet', f) for f in os.listdir('dataSet')]
    faces = []
    ids = []

    for caminhoImagem in caminhos:

        # Converter em escala de cinzento
        imagemFace = Image.open(caminhoImagem).convert('L')

        # Redimensionamento das imagens
        imagemFace = imagemFace.resize((220, 220))

        # Converter a imagem em array de NP
        imagemNP = np.array(imagemFace, 'uint8')

        # Split do nome dos arquivos
        id = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))
        ids.append(id)
        faces.append(imagemNP)

    window.labelText.setText("A treinar...")

    Y = np.array(ids)
    X = np.array(faces)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

    # Realizar o treino e gravação do Classificador
    lbph.train(X_train, Y_train)
    lbph.write('classificadorLBPH.yml')

    window.labelText.setText("Treinamento Finalizado")

    # Obtenção do Y_predict
    Y_predict = np.zeros((len(Y_test),), dtype=int)
    for i in range(len(X_test)):
        a = lbph.predict(X_test[i, :, :])
        Y_predict[i] = a[0]

    # Obtenção da matriz de confusão
    cm = metrics.confusion_matrix(Y_test, Y_predict, labels=[1, 2, 3, 4, 5, 6, 7, 8])
    window.console.setText("Confusion Matrix:")
    window.console.append(str(cm))

    # Obtenção do Classification Report
    cr = metrics.classification_report(Y_test, Y_predict)
    window.console.append("Classification Report:")
    window.console.append(str(cr))
    #print (cr)

    # Matriz de confusão em forma de gráfico
    sum = cm.sum()
    df_cm = pd.DataFrame(cm,
                         index=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],
                         columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'])

    res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.show()


camera = cv2.VideoCapture(0)
app = QtWidgets.QApplication(sys.argv)

# Carregamento do arquivo *.ui
window = uic.loadUi("mainWindow.ui")

# Função de cada botão
window.botaoCameraOn.clicked.connect(on_cameraON_clicked)
window.botaoRecImagem.clicked.connect(RecImagem_clicked)
window.botaoSair.clicked.connect(sair_clicked)
window.botaoTreinamento.clicked.connect(treinamento)
window.ButtonStart.clicked.connect(start_clicked)
window.ButtonStop.clicked.connect(stop_clicked)

window.show()
app.exec()
