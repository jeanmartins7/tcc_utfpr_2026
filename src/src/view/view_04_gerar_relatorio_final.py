import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
import argparse

IMG_TAM = 256

def carregar_modelo(caminho):
    if os.path.exists(caminho):
        try:
            print(f"   [OK] Carregando: {os.path.basename(caminho)}")
            return tf.keras.models.load_model(caminho)
        except Exception as e:
            print(f"   [ERRO] Falha ao abrir {caminho}: {e}")
            return None
    else:
        print(f"   [AVISO] Modelo não encontrado: {caminho}")
        return None

def gerar_relatorio(args):
    print(f"\n--- GERANDO RELATÓRIO (Flow x{args.amplification}) ---")
    
    path_base = os.path.join(args.results, "exp1_baseline", "modelo_baseline_unet.h5")
    path_aug  = os.path.join(args.results, "exp2_augmentation", "modelo_exp2_aug.h5")
    path_flow = os.path.join(args.results, "exp3_flow_architecture", "modelo_exp3_flow.h5")

    model_base = carregar_modelo(path_base)
    model_aug  = carregar_modelo(path_aug)
    model_flow = carregar_modelo(path_flow)

    pasta_val_img = os.path.join(args.dataset, "val", "images")
    pasta_val_mask = os.path.join(args.dataset, "val", "masks")
    
    if not os.path.exists(pasta_val_img):
        print(f"ERRO: Validação não encontrada em {pasta_val_img}")
        return

    arquivos = sorted(glob.glob(os.path.join(pasta_val_img, "*.jpg")))
    if not arquivos: return

    qtd_amostras = min(3, len(arquivos))
    amostras = random.sample(arquivos, qtd_amostras)

    fig, axes = plt.subplots(qtd_amostras, 5, figsize=(20, 4 * qtd_amostras))
    fig.suptitle(f"Comparativo Final - Dataset: {os.path.basename(args.dataset)}", fontsize=16)

    for i, caminho_img in enumerate(amostras):
        img_rgb = cv2.imread(caminho_img)
        img_rgb = cv2.resize(img_rgb, (IMG_TAM, IMG_TAM))
        input_rgb = img_rgb / 255.0
        input_rgb_batch = np.expand_dims(input_rgb, axis=0)

        nome = os.path.basename(caminho_img)
        caminho_mask = os.path.join(pasta_val_mask, nome)
        if os.path.exists(caminho_mask):
            mask = cv2.imread(caminho_mask, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_TAM, IMG_TAM))
        else:
            mask = np.zeros((IMG_TAM, IMG_TAM), dtype=np.uint8)

        nome_flow = nome.replace("frame", "flow")
        caminho_flow = os.path.join(args.flow, nome_flow)
        
        if os.path.exists(caminho_flow):
            flow_img = cv2.imread(caminho_flow)
            flow_img = cv2.resize(flow_img, (IMG_TAM, IMG_TAM))
            flow_input = (flow_img / 255.0) * args.amplification
        else:
            flow_input = np.zeros((IMG_TAM, IMG_TAM, 3))

        input_hibrido = np.concatenate([input_rgb, flow_input], axis=-1)
        input_hibrido_batch = np.expand_dims(input_hibrido, axis=0)

        pred_base = np.zeros((IMG_TAM, IMG_TAM))
        if model_base:
            p = model_base.predict(input_rgb_batch, verbose=0)[0]
            pred_base = (p > 0.5).astype(np.uint8) * 255
            pred_base = pred_base.reshape(IMG_TAM, IMG_TAM)

        pred_aug = np.zeros((IMG_TAM, IMG_TAM))
        if model_aug:
            p = model_aug.predict(input_rgb_batch, verbose=0)[0]
            pred_aug = (p > 0.5).astype(np.uint8) * 255
            pred_aug = pred_aug.reshape(IMG_TAM, IMG_TAM)

        pred_flow = np.zeros((IMG_TAM, IMG_TAM))
        if model_flow:
            p = model_flow.predict(input_hibrido_batch, verbose=0)[0]
            pred_flow = (p * 255).astype(np.uint8) 
            pred_flow = pred_flow.reshape(IMG_TAM, IMG_TAM)

        ax = axes[i] if qtd_amostras > 1 else axes
        
        ax[0].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Gabarito")
        ax[1].axis('off')

        ax[2].imshow(pred_base, cmap='gray', vmin=0, vmax=255)
        ax[2].set_title("Exp 1: Baseline")
        ax[2].axis('off')

        ax[3].imshow(pred_aug, cmap='gray', vmin=0, vmax=255)
        ax[3].set_title("Exp 2: Augmentation")
        ax[3].axis('off')

        ax[4].imshow(pred_flow, cmap='gray', vmin=0, vmax=255)
        ax[4].set_title(f"Exp 3: Flow (x{args.amplification})")
        ax[4].axis('off')

    plt.tight_layout()
    caminho_final = os.path.join(args.results, args.output)
    plt.savefig(caminho_final)
    print(f"\n✅ Relatório salvo em: {os.path.abspath(caminho_final)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-f", "--flow", required=True)
    parser.add_argument("-r", "--results", required=True)
    parser.add_argument("-o", "--output", default="RELATORIO_FINAL.png")
    parser.add_argument("-a", "--amplification", type=float, default=5.0, help="Fator usado no treino")
    
    args = parser.parse_args()
    gerar_relatorio(args)