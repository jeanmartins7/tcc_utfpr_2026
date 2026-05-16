import os
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import glob
import shutil

def criar_estrutura_pastas(pasta_base):
    """Cria a árvore de diretórios train/val."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(pasta_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(pasta_base, split, 'masks'), exist_ok=True)

def processar_e_salvar(lista_arquivos, pasta_origem_masks, pasta_destino, split_nome, size):
    """Lê, redimensiona e salva imagens e máscaras."""
    print(f"   -> Processando {len(lista_arquivos)} imagens para '{split_nome}'...")
    
    largura, altura = size

    for caminho_img in lista_arquivos:
        nome_arq = os.path.basename(caminho_img)
        img = cv2.imread(caminho_img)
        
        if img is None:
            print(f"Aviso: Não foi possível ler {caminho_img}")
            continue

        img = cv2.resize(img, (largura, altura))

        caminho_dest_img = os.path.join(pasta_destino, split_nome, 'images', nome_arq)
        cv2.imwrite(caminho_dest_img, img)

        caminho_mask_origem = os.path.join(pasta_origem_masks, nome_arq)

        if os.path.exists(caminho_mask_origem):
            mask = cv2.imread(caminho_mask_origem, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (largura, altura))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = np.zeros((altura, largura), dtype=np.uint8)
            print(f"Aviso: Máscara não encontrada para {nome_arq}. Criando vazia.")

        caminho_dest_mask = os.path.join(pasta_destino, split_nome, 'masks', nome_arq)
        cv2.imwrite(caminho_dest_mask, mask)

def montar_dataset(frames_dir, masks_dir, output_dir, width, height, val_size):
    if not os.path.exists(frames_dir):
        print(f"ERRO: Pasta de frames não encontrada: {frames_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"ERRO: Pasta de máscaras não encontrada: {masks_dir}")
        return

    if os.path.exists(output_dir):
        print(f"Aviso: A pasta de destino '{output_dir}' já existe. Mesclando/Sobrescrevendo...")
    else:
        print(f"Criando pasta de destino: {output_dir}")
    
    criar_estrutura_pastas(output_dir)

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Total de frames encontrados: {len(frames)}")
    
    if len(frames) == 0:
        print("Abortando: Nenhum arquivo .jpg encontrado.")
        return

    train_frames, val_frames = train_test_split(frames, test_size=val_size, random_state=42)

    size = (width, height)
    processar_e_salvar(train_frames, masks_dir, output_dir, 'train', size)
    processar_e_salvar(val_frames, masks_dir, output_dir, 'val', size)

    print(f"\n✅ DATASET PRONTO EM: {os.path.abspath(output_dir)}")
    print(f"   Treino: {len(train_frames)} imagens")
    print(f"   Validação: {len(val_frames)} imagens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organiza, redimensiona e divide o dataset.")
    
    parser.add_argument("-f", "--frames", required=True, help="Pasta de origem dos Frames (RGB)")
    parser.add_argument("-m", "--masks", required=True, help="Pasta de origem das Máscaras (P&B)")
    parser.add_argument("-o", "--output", required=True, help="Pasta de destino do Dataset Pronto")
    parser.add_argument("--width", type=int, default=256, help="Largura final (Padrão: 256)")
    parser.add_argument("--height", type=int, default=256, help="Altura final (Padrão: 256)")
    parser.add_argument("--split", type=float, default=0.2, help="Porcentagem de validação (Padrão: 0.2 = 20%)")

    args = parser.parse_args()

    montar_dataset(args.frames, args.masks, args.output, args.width, args.height, args.split)