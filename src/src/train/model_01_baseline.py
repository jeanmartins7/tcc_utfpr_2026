import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import glob
import argparse

def carregar_dados(pasta_dataset, tipo_split, width, height):
    pasta_imgs = os.path.join(pasta_dataset, tipo_split, "images")
    pasta_masks = os.path.join(pasta_dataset, tipo_split, "masks")
    
    if not os.path.exists(pasta_imgs):
        print(f"ERRO: Pasta não encontrada: {pasta_imgs}")
        return np.array([]), np.array([])

    arquivos = sorted(glob.glob(os.path.join(pasta_imgs, "*.jpg")))
    X, Y = [], []
    
    print(f"Carregando {tipo_split} ({len(arquivos)} arquivos)...")
    
    for arq in arquivos:
        img = cv2.imread(arq)
        img = cv2.resize(img, (width, height)) / 255.0
        X.append(img)
        
        nome = os.path.basename(arq)
        mask = cv2.imread(os.path.join(pasta_masks, nome), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height)) / 255.0
        Y.append(np.expand_dims(mask, axis=-1))
        
    return np.array(X), np.array(Y)

def construir_unet(width, height, channels):
    inputs = layers.Input(shape=(height, width, channels))

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

    X_train, Y_train = carregar_dados(args.dataset, 'train', args.width, args.height)
    X_val, Y_val = carregar_dados(args.dataset, 'val', args.width, args.height)

    if len(X_train) == 0:
        print("Abortando: Sem dados de treino.")
        return

    model = construir_unet(args.width, args.height, 3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"--- INICIANDO BASELINE ({args.epochs} épocas) ---")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                        batch_size=args.batch, epochs=args.epochs)

    model.save(os.path.join(args.output, "modelo_baseline_unet.h5"))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino Acc')
    plt.plot(history.history['val_accuracy'], label='Validação Acc')
    plt.title('Acurácia (Baseline)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino Loss')
    plt.plot(history.history['val_loss'], label='Validação Loss')
    plt.title('Loss (Baseline)')
    plt.legend()
    plt.savefig(os.path.join(args.output, "grafico_baseline.png"))
    print("Baseline Concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Pasta do Dataset Pronto")
    parser.add_argument("-o", "--output", required=True, help="Pasta para salvar modelo/grafico")
    parser.add_argument("-e", "--epochs", type=int, default=15, help="Qtd de Épocas")
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch Size")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    treinar(args)