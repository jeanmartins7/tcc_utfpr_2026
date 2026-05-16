import cv2
import numpy as np
import os
import glob
import argparse

def aplicar_realce_contraste(imagem_cinza):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(imagem_cinza)

def gerar_fluxo_optico_denso(diretorio_frames, diretorio_saida, limiar, step):
    arquivos = sorted(glob.glob(os.path.join(diretorio_frames, "*.jpg")))
    if len(arquivos) < step + 1:
        print(f"ERRO: Menos de {step+1} frames. Impossível aplicar passo {step}.")
        return

    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    print(f"--- Gerando Fluxo (Passo={step}, Limiar={limiar}) ---")

    max_movimento_global = 0

    total = len(arquivos) - step 
    
    for i in range(total):
        frame_anterior = cv2.imread(arquivos[i])
        frame_atual = cv2.imread(arquivos[i + step])

        cinza_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
        cinza_atual = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2GRAY)

        cinza_anterior = aplicar_realce_contraste(cinza_anterior)
        cinza_atual = aplicar_realce_contraste(cinza_atual)

        fluxo = cv2.calcOpticalFlowFarneback(
            prev=cinza_anterior, 
            next=cinza_atual, 
            flow=None,
            pyr_scale=0.5, levels=3, winsize=10, 
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        magnitude, angulo = cv2.cartToPolar(fluxo[..., 0], fluxo[..., 1])

        max_mag_frame = np.max(magnitude)
        if max_mag_frame > max_movimento_global:
            max_movimento_global = max_mag_frame

        mask_ruido = magnitude < limiar 
        magnitude[mask_ruido] = 0
        angulo[mask_ruido] = 0

        hsv = np.zeros_like(frame_anterior)
        hsv[..., 1] = 255
        hsv[..., 0] = angulo * 180 / np.pi / 2

        if max_mag_frame > 0:
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 2] = 0
        
        fluxo_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        nome_original = os.path.basename(arquivos[i])
        nome_arquivo = nome_original.replace("frame", "flow")
        
        cv2.imwrite(os.path.join(diretorio_saida, nome_arquivo), fluxo_bgr)

        if i % 100 == 0:
            print(f"Par {i}/{total} (Frame {i} vs {i+step}) | Max Mov: {max_mag_frame:.2f} px")

    print(f"\n✅ CONCLUÍDO!")
    print(f"Maior movimento registrado (com passo {step}): {max_movimento_global:.2f} pixels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Pasta frames RGB")
    parser.add_argument("-o", "--output", required=True, help="Pasta destino Fluxo")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Limiar ruído")
    parser.add_argument("-s", "--step", type=int, default=1, help="Pulo entre frames (Aumente se o movimento for lento)")
    
    args = parser.parse_args()
    
    gerar_fluxo_optico_denso(args.input, args.output, args.threshold, args.step)