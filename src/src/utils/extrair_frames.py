#!/usr/bin/env python3

import cv2
import os
import argparse

def extrair_frames(caminho_video, pasta_saida, largura=None, altura=None):
    if not os.path.exists(caminho_video):
        print(f"ERRO: O arquivo '{caminho_video}' não foi encontrado.")
        return

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        print(f"Pasta '{pasta_saida}' criada.")

    cap = cv2.VideoCapture(caminho_video)

    if not cap.isOpened():
        print("ERRO: Não foi possível abrir o vídeo (formato inválido ou corrompido).")
        return

    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"--- Iniciando extração de {caminho_video} ---")
    print(f"Total estimado de frames: {total_frames}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if largura and altura:
            frame = cv2.resize(frame, (largura, altura))

        nome_arquivo = os.path.join(pasta_saida, f"frame_{count:05d}.jpg")
        cv2.imwrite(nome_arquivo, frame)

        if count % 50 == 0:
            print(f"Processado: {count}/{total_frames} frames...")
        
        count += 1

    cap.release()
    print(f"\nSUCESSO! {count} frames salvos em '{pasta_saida}'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extrai frames de um vídeo para criar dataset.")

    parser.add_argument("-v", "--video", required=True, help="Caminho do arquivo de vídeo.")

    parser.add_argument("-o", "--output", required=True, help="Pasta onde os frames serão salvos.")

    parser.add_argument("--width", type=int, default=None, help="Nova largura (opcional).")
    parser.add_argument("--height", type=int, default=None, help="Nova altura (opcional).")

    args = parser.parse_args()

    extrair_frames(args.video, args.output, args.width, args.height)