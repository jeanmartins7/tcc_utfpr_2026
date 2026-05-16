import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import argparse
import matplotlib.pyplot as plt

def carregar_dados_hibridos(pasta_dataset, pasta_flow, tipo_split, width, height, amplification):
    pasta_imgs = os.path.join(pasta_dataset, tipo_split, "images")
    pasta_masks = os.path.join(pasta_dataset, tipo_split, "masks")
    
    arquivos = sorted(glob.glob(os.path.join(pasta_imgs, "*.jpg")))
    X, Y = [], []
    
    print(f"Carregando {tipo_split} com FUSÃO (Amplificação = {amplification}x)...")
    
    for arq in arquivos:
        img = cv2.imread(arq)
        img = cv2.resize(img, (width, height)) / 255.0

        nome = os.path.basename(arq)
        nome_flow = nome.replace("frame", "flow")
        caminho_flow = os.path.join(pasta_flow, nome_flow)
        
        if os.path.exists(caminho_flow):
            flow = cv2.imread(caminho_flow)
            flow = cv2.resize(flow, (width, height)) 
            # --- O PULO DO GATO: AMPLIFICAÇÃO PARAMETRIZADA ---
            flow = (flow / 255.0) * amplification
        else:
            flow = np.zeros_like(img)

        input_hibrido = np.concatenate([img, flow], axis=-1)
        X.append(input_hibrido)

        mask = cv2.imread(os.path.join(pasta_masks, nome), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height)) / 255.0
        Y.append(np.expand_dims(mask, axis=-1))
        
    return np.array(X), np.array(Y)

def construir_unet_6_canais(width, height):
    inputs = layers.Input(shape=(height, width, 6))

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    
    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    return models.Model(inputs=[inputs], outputs=[outputs])

def treinar(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    X_train, Y_train = carregar_dados_hibridos(
        args.dataset, args.flow, 'train', args.width, args.height, args.amplification
    )
    X_val, Y_val = carregar_dados_hibridos(
        args.dataset, args.flow, 'val', args.width, args.height, args.amplification
    )

    model = construir_unet_6_canais(args.width, args.height)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"--- INICIANDO EXP 3 (FLUXO - {args.epochs} épocas) ---")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                        batch_size=args.batch, epochs=args.epochs)

    model.save(os.path.join(args.output, "modelo_exp3_flow.h5"))

    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Treino Acc')
        plt.plot(history.history['val_accuracy'], label='Validação Acc')
        plt.title(f'Acurácia (Flow x{args.amplification})')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Treino Loss')
        plt.plot(history.history['val_loss'], label='Validação Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output, "grafico_exp3_flow.png"))
        print("Gráfico salvo com sucesso.")
    except Exception as e:
        print(f"Aviso: Não foi possível gerar o gráfico ({e}), mas o modelo foi salvo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-f", "--flow", required=True, help="Pasta com imagens de Fluxo")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=15)
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("-a", "--amplification", type=float, default=5.0, help="Fator de multiplicação do fluxo (Default: 5.0)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    treinar(args)