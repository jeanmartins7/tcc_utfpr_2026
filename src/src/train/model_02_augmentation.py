import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob
import argparse

def carregar_dados(pasta_dataset, tipo_split, width, height):
    # (Reaproveitando a lógica de carregamento padrão)
    pasta_imgs = os.path.join(pasta_dataset, tipo_split, "images")
    pasta_masks = os.path.join(pasta_dataset, tipo_split, "masks")
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

def criar_geradores(X, Y, batch_size):
    data_gen_args = dict(
        rotation_range=90, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    image_datagen.fit(X, augment=True, seed=seed)
    mask_datagen.fit(Y, augment=True, seed=seed)
    
    image_gen = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_gen = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)
    
    def gerador_combinado(gen_x, gen_y):
        while True:
            yield (next(gen_x), next(gen_y))

    return gerador_combinado(image_gen, mask_gen)

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

    train_gen = criar_geradores(X_train, Y_train, args.batch)
    
    model = construir_unet(args.width, args.height, 3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"--- INICIANDO EXP 2 (AUGMENTATION - {args.epochs} épocas) ---")
    history = model.fit(
        train_gen,
        validation_data=(X_val, Y_val),
        steps_per_epoch=len(X_train) // args.batch,
        epochs=args.epochs
    )

    model.save(os.path.join(args.output, "modelo_exp2_aug.h5"))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino Acc')
    plt.plot(history.history['val_accuracy'], label='Validação Acc')
    plt.title('Acurácia (Augmentation)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino Loss')
    plt.plot(history.history['val_loss'], label='Validação Loss')
    plt.title('Loss (Augmentation)')
    plt.legend()
    plt.savefig(os.path.join(args.output, "grafico_exp2_aug.png"))
    print("Exp 2 Concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=25)
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    treinar(args)