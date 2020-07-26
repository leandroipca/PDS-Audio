import os

def teste():

    caminhos = [os.path.join('dataSetAudio/comandos/testes', f) for f in os.listdir('dataSetAudio/comandos/testes')]
    x = []
    y = []

    for caminhoAudio in caminhos:


        # Split do nome dos arquivos
        id = str(os.path.split(caminhoAudio)[1].split('_')[0])
        x.append(id)

        #y.append(imagemNP)

x = teste()
print (x)