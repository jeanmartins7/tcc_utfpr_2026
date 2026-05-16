# Segmentação de Objetos Camuflados com Fluxo Óptico Denso
Este projeto propõe uma abordagem de Deep Learning Híbrido para segmentar objetos pequenos e camuflados (mimetismo) em vídeos. A arquitetura combina redes neurais convolucionais (U-Net) com engenharia de dados temporal (Fluxo Óptico), superando as limitações de abordagens estáticas baseadas apenas em cor.

(Exemplo de resultado no dataset camuflado: Original vs. Baseline vs. Fluxo Óptico)

## 📋 Sobre o Projeto
A detecção de objetos camuflados é um desafio clássico em Visão Computacional. Modelos tradicionais falham devido ao histograma de cor idêntico entre objeto e fundo.

Este trabalho implementa um pipeline completo de três experimentos:

1. Baseline (RGB): U-Net padrão treinada em frames estáticos. (Falha esperada na camuflagem).

2. Melhoria 1 (Augmentation): U-Net com aumento de dados agressivo.

3. Melhoria 2 (Híbrido): Nova topologia de rede aceitando 6 canais (RGB + Fluxo Óptico), utilizando o movimento relativo para quebrar a camuflagem.

## 🛠️ Tecnologias Utilizadas
- Linguagem: Python 3.11

- Deep Learning: TensorFlow / Keras

- Visão Computacional: OpenCV (cv2)

- Processamento de Dados: NumPy, Scikit-learn

- Visualização: Matplotlib

- Hardware: Suporte nativo a GPU NVIDIA (CUDA 12 via tensorflow[and-cuda])

``` Plaintext
.
├── dataset/                 # Pasta onde ficam os vídeos e dados gerados
│   ├── video/               # Coloque seus arquivos .mp4 ou .MOV aqui
│   ├── frames/              # Frames extraídos
│   ├── masks/               # Gabaritos (Ground Truth)
│   └── flow/                # Imagens de Fluxo Óptico calculadas
│
├── result/                  # Resultados
    .
    ├── black_experiment
    │   ├── exp1_baseline
    │   │   ├── grafico_baseline.png
    │   │   └── modelo_baseline_unet.h5
    │   ├── exp2_augmentation
    │   │   ├── grafico_exp2_aug.png
    │   │   └── modelo_exp2_aug.h5
    │   ├── exp3_flow_architecture
    │   │   ├── grafico_exp3_flow.png
    │   │   └── modelo_exp3_flow.h5
    │   ├── RELATORIO_BLACK.png
    │   ├── RELATORIO_BLACK_v1.png
    │   ├── RELATORIO_BLACK_V2.png
    │   ├── RELATORIO_BLACK_V3.png
    │   ├── RELATORIO_BLACK_V4.png
    │   ├── RELATORIO_BLACK_V5.png
    │   ├── RELATORIO_BLACK_V6.png
    │   ├── RELATORIO_BLACK_V7.png
    │   ├── RELATORIO_BLACK_V8.png
    │   └── RELATORIO_BLACK_V9.png
    ├── green_experiment
    │   ├── exp1_baseline
    │   │   ├── grafico_baseline.png
    │   │   └── modelo_baseline_unet.h5
    │   ├── exp2_augmentation
    │   │   ├── grafico_exp2_aug.png
    │   │   └── modelo_exp2_aug.h5
    │   ├── exp3_flow_architecture
    │   │   ├── grafico_exp3_flow.png
    │   │   └── modelo_exp3_flow.h5
    │   ├── RELATORIO_FINAL.png
    │   ├── RELATORIO_FINAL_V10.png
    │   ├── RELATORIO_FINAL_V1.png
    │   ├── RELATORIO_FINAL_V2.png
    │   ├── RELATORIO_FINAL_V3.png
    │   ├── RELATORIO_FINAL_V4.png
    │   ├── RELATORIO_FINAL_V5.png
    │   ├── RELATORIO_FINAL_V6.png
    │   ├── RELATORIO_FINAL_V7.png
    │   ├── RELATORIO_FINAL_V8.png
    │   └── RELATORIO_FINAL_V9.png
    └── red_experiment
        ├── exp1_baseline
        │   ├── grafico_baseline.png
        │   └── modelo_baseline_unet.h5
        ├── exp2_augmentation
        │   ├── grafico_exp2_aug.png
        │   └── modelo_exp2_aug.h5
        ├── exp3_flow_architecture
        │   ├── grafico_exp3_flow.png
        │   └── modelo_exp3_flow.h5
        └── RELATORIO_RED_V3.png
    │
├── src/                     # Código Fonte
    .
    ├── preparation
    │   ├── data_01_gerar_labels_cor.py
    │   ├── data_02_gerar_labels_movimento.py
    │   └── data_03_montar_dataset_final.py
    ├── train
    │   ├── model_01_baseline.py
    │   ├── model_02_augmentation.py
    │   ├── model_03_flow_hibrido.py
    │   └── os
    ├── utils
    │   ├── argparse
    │   ├── calcular_fluxo.py
    │   ├── cv2
    │   ├── extrair_frames.py
    │   └── os
    └── view
        ├── view_04_gerar_relatorio_final.py
        └── visualizar_resultados.py
└── requirements.txt         # Dependências do Python
``` 

## 🚀 Como Executar (Passo a Passo)
1. Pré-requisitos
Certifique-se de ter o Python 3.11 instalado.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-projeto.git
cd seu-projeto

# Crie e ative o ambiente virtual
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
# .\venv\Scripts\activate # Windows

# Instale as dependências
pip install -r requirements.txt
```

(Se você usa Linux com GPU NVIDIA, certifique-se de configurar as variáveis de ambiente do CUDA conforme a documentação do TensorFlow).

2. Coloque seus Vídeos
Salve seus vídeos de teste na pasta dataset/video/:

test_1.MOV (Vídeo do inseto camuflado/preto)

test_2.MOV (Vídeo do inseto camuflado/vermelho)

test_3.MOV (Vídeo do inseto camuflado/verde)

3. Execução Manual (Passo a Passo)
Se preferir rodar etapa por etapa para entender o processo:

Fase 1: Preparação dos Dados

```bash
# 1. Extrair Frames do vídeo
python3 src/extrair_frames.py
-v ../../dataset/video/test_3.MOV
-o ../../dataset/frames/frames_video_green

# 2. Gerar Gabarito (Ground Truth) via Movimento
python3 src/data_02_gerar_labels_movimento.py
-i ../../dataset/frames/frames_video_green
-o ../../dataset/masks/masks_green

# 3. Calcular Fluxo Óptico Turbo (Step 5, Threshold 0.5)
python3 src/calcular_fluxo.py
-i ../../dataset/frames/frames_video_green
-o dataset/flow/flow_video-green -t 0.5 -s 15

# 4. Organizar Dataset Final
python3 src/data_03_montar_dataset_final.py
-f ../../dataset/frames/frames_video_green
-m ../../dataset/masks/masks_video_green
-o ../../dataset_result/green
```

Fase 2: Treinamento
```bash

# Treinar Modelo baseline (Net-U)
python3 model_01_baseline.py -d ../../dataset_result/green -o ../../result/green_experiment/exp1_baseline -e 50 

# Treinar Modelo Augmentation
python3 model_02_augmentation.py -d ../../dataset_result/green -o ../../result/green_experiment/exp2_augmentation -e 50 

# Treinar Modelo Híbrido (RGB + Fluxo)
python3 model_03_flow_hibrido.py -d ../../dataset_result/green -f ../../dataset/flow/flow_video-green -o ../../result/green_experiment/exp3_flow_architecture -e 50 -a 10.0
```

Fase 3: Visualização
```bash
python3 view_04_gerar_relatorio_final.py -d ../../dataset_result/green -f ../../dataset/flow/flow_video-green -r ../../result/green_experiment -o RELATORIO_FINAL.png -a 10.0 
```