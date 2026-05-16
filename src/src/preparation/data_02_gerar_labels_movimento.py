import cv2
import numpy as np
import os
import glob
import argparse

LIMIAR_MOVIMENTO = 25
AREA_MINIMA = 10

def gerar_mascaras(pasta_frames, pasta_saida):
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        print(f"Pasta criada: {pasta_saida}")

    arquivos = sorted(glob.glob(os.path.join(pasta_frames, "*.jpg")))
    
    if len(arquivos) < 2:
        print("ERRO: Preciso de pelo menos 2 frames para detectar movimento!")
        return

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=LIMIAR_MOVIMENTO, detectShadows=False)

    print(f"--- Gerando Máscaras para {len(arquivos)} frames ---")
    print(f"Origem: {pasta_frames}")
    print(f"Destino: {pasta_saida}")

    for i, caminho in enumerate(arquivos):
        frame = cv2.imread(caminho)
        if frame is None: continue

        mask = backSub.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_limpa = np.zeros_like(mask)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > AREA_MINIMA:
                cv2.drawContours(mask_limpa, [cnt], -1, 255, -1)

        nome_arq = os.path.basename(caminho)
        caminho_final = os.path.join(pasta_saida, nome_arq)
        cv2.imwrite(caminho_final, mask_limpa)

        if i % 100 == 0:
            print(f"Processado {i}/{len(arquivos)}...")

    print("Concluído! Máscaras geradas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera máscaras binárias baseadas em movimento.")
    parser.add_argument("-i", "--input", required=True, help="Pasta dos frames originais (RGB)")
    parser.add_argument("-o", "--output", required=True, help="Pasta onde salvar as máscaras (P&B)")
    
    args = parser.parse_args()
    
    gerar_mascaras(args.input, args.output)