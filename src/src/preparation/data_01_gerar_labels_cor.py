#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import os
import argparse

PASTA_FRAMES = "../../dataset/frames/frames_video_2"
PASTA_SAIDA = "../../dataset/masks/masks_video_2"

def nada(x):
    pass

def criar_ferramenta_hsv():

    arquivos = sorted(glob.glob(os.path.join(PASTA_FRAMES, "*.jpg")))
    if not arquivos:
        print(f"ERRO: Nenhuma imagem encontrada em {PASTA_FRAMES}")
        return

    indice_img = 50 if len(arquivos) > 50 else 0
    img_calibracao = cv2.imread(arquivos[indice_img])

    img_calibracao = cv2.resize(img_calibracao, (640, 640)) 

    cv2.namedWindow("Calibrador de Cor (Aperte 'S' para salvar e processar)")

    cv2.createTrackbar("H Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 0, 179, nada)
    cv2.createTrackbar("H Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 179, 179, nada)
    cv2.createTrackbar("S Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 50, 255, nada)
    cv2.createTrackbar("S Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 255, 255, nada)
    cv2.createTrackbar("V Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 50, 255, nada)
    cv2.createTrackbar("V Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)", 255, 255, nada)

    print("--- INSTRUÇÕES ---")
    print("1. Ajuste as barras até o OBJETO ficar BRANCO e o fundo PRETO.")
    print("2. Tente remover o máximo de ruído do fundo.")
    print("3. Quando estiver bom, aperte a tecla 's' no teclado para gerar todas as máscaras.")
    print("4. Aperte 'q' para sair sem salvar.")

    h_min, h_max, s_min, s_max, v_min, v_max = 0, 179, 0, 255, 0, 255

    while True:
        h_min = cv2.getTrackbarPos("H Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)")
        h_max = cv2.getTrackbarPos("H Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)")
        s_min = cv2.getTrackbarPos("S Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)")
        s_max = cv2.getTrackbarPos("S Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)")
        v_min = cv2.getTrackbarPos("V Min", "Calibrador de Cor (Aperte 'S' para salvar e processar)")
        v_max = cv2.getTrackbarPos("V Max", "Calibrador de Cor (Aperte 'S' para salvar e processar)")

        hsv = cv2.cvtColor(img_calibracao, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)

        resultado = cv2.bitwise_and(img_calibracao, img_calibracao, mask=mask)
        vis = np.hstack((img_calibracao, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), resultado))

        vis_pequeno = cv2.resize(vis, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow("Calibrador de Cor (Aperte 'S' para salvar e processar)", vis_pequeno)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"\nValores escolhidos: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
            print("Iniciando processamento em lote...")
            cv2.destroyAllWindows()
            processar_todas(lower, upper, arquivos)
            break

    cv2.destroyAllWindows()

def processar_todas(lower, upper, arquivos):
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    total = len(arquivos)
    for i, caminho in enumerate(arquivos):
        img = cv2.imread(caminho)

        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        nome_arquivo = os.path.basename(caminho)
        caminho_final = os.path.join(PASTA_SAIDA, nome_arquivo)
        cv2.imwrite(caminho_final, mask)

        if i % 100 == 0:
            print(f"Gerando máscara {i}/{total}...")

    print(f"\nSUCESSO! {total} máscaras geradas em '{PASTA_SAIDA}'")

if __name__ == "__main__":
    criar_ferramenta_hsv()