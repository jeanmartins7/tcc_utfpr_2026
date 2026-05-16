import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random

PASTA_DATASET = "../../dataset_red_pronto"
PASTA_FLOW = "../../dataset/flow/flow_video_red"
PASTA_RESULTADOS = "../../result/red_experiment/exp3_flow_architecture"
MODELO_PATH = os.path.join(PASTA_RESULTADOS, "modelo_exp3_flow.h5")
IMG_TAM = 256

def visualizar_hibrido():
    print("\n=== VISUALIZADOR V3 - MODO SENSÍVEL ===")
    
    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Modelo não encontrado em {MODELO_PATH}")
        return
        
    print("Carregando modelo...")
    try:
        model = tf.keras.models.load_model(MODELO_PATH)
    except Exception as e:
        print(f"Erro no load_model: {e}")
        return

    pasta_val_img = os.path.join(PASTA_DATASET, "val", "images")
    pasta_val_mask = os.path.join(PASTA_DATASET, "val", "masks")
    arquivos = sorted(glob.glob(os.path.join(pasta_val_img, "*.jpg")))
    
    if not arquivos: return

    amostras = random.sample(arquivos, min(3, len(arquivos)))
    plt.figure(figsize=(12, 12))

    for i, caminho_img in enumerate(amostras):
        img_rgb = cv2.imread(caminho_img)
        img_rgb = cv2.resize(img_rgb, (IMG_TAM, IMG_TAM))
        input_rgb = img_rgb / 255.0

        nome = os.path.basename(caminho_img)
        nome_flow = nome.replace("frame", "flow")
        caminho_flow = os.path.join(PASTA_FLOW, nome_flow)
        
        img_flow = np.zeros((IMG_TAM, IMG_TAM, 3), dtype=np.uint8)
        if os.path.exists(caminho_flow):
            temp_flow = cv2.imread(caminho_flow)
            if temp_flow is not None:
                img_flow = cv2.resize(temp_flow, (IMG_TAM, IMG_TAM))

        input_flow = (img_flow / 255.0) * 5.0

        input_hibrido = np.concatenate([input_rgb, input_flow], axis=-1)
        input_hibrido = np.expand_dims(input_hibrido, axis=0)

        predicao = model.predict(input_hibrido, verbose=0)[0]
        predicao = (predicao * 255).astype(np.uint8)
        predicao = predicao.reshape(IMG_TAM, IMG_TAM)

        caminho_mask = os.path.join(pasta_val_mask, nome)
        if os.path.exists(caminho_mask):
            mask_real = cv2.imread(caminho_mask, cv2.IMREAD_GRAYSCALE)
            mask_real = cv2.resize(mask_real, (IMG_TAM, IMG_TAM))
        else:
            mask_real = np.zeros((IMG_TAM, IMG_TAM), dtype=np.uint8)

        plt.subplot(3, 4, i*4 + 1)
        plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, 4, i*4 + 2)
        plt.imshow(cv2.cvtColor(img_flow, cv2.COLOR_BGR2RGB))
        plt.title("Fluxo")
        plt.axis('off')

        plt.subplot(3, 4, i*4 + 3)
        plt.imshow(mask_real, cmap='gray')
        plt.title("Gabarito")
        plt.axis('off')

        plt.subplot(3, 4, i*4 + 4)
        plt.imshow(predicao, cmap='gray')
        plt.title("Predição (Probabilidade)")
        plt.axis('off')

    plt.tight_layout()
    caminho_salvar = os.path.join(PASTA_RESULTADOS, "resultado_visual_exp3.png")
    plt.savefig(caminho_salvar)
    print(f"\n✅ Imagem salva em: {caminho_salvar}")

if __name__ == "__main__":
    visualizar_hibrido()