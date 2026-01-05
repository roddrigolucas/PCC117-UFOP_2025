Uma extensão híbrida do ORB-SLAM integrando redes neurais profundas para SLAM visual robusto em robôs móveis


📋 Resumo

O Neural ORB-SLAM é uma arquitetura híbrida inovadora que combina a robustez de redes neurais profundas com a precisão da otimização geométrica clássica do ORB-SLAM. O sistema substitui componentes tradicionais por módulos de deep learning estado-da-arte:
ComponenteMétodo OriginalMétodo NeuralExtração de FeaturesORBSuperPointFeature MatchingForça BrutaSuperGlueEstimação de ProfundidadeN/AMiDaS v3.1Filtragem DinâmicaRANSACYOLOv8Loop ClosingDBoW2NetVLAD

🎯 Resultados Principais

Avaliado no benchmark KITTI Odometry:
MétricaORB-SLAM2ORB-SLAM3DROID-SLAMNeural ORB-SLAMATE (m) ↓15.4211.876.238.91Taxa Tracking ↑74.3%82.1%99.2%91.8%FPS ↑31.229.88.418.3

✅ Melhorias Alcançadas

📉 42.2% de redução no erro de trajetória vs ORB-SLAM2
📈 23.7% de melhoria na taxa de tracking
⚡ 2.18× mais rápido que DROID-SLAM
🌙 8.3% de degradação com variação de iluminação (vs 34.2% do ORB-SLAM2)


🏗️ Arquitetura
                         ┌─────────────┐
                         │  Imagem It  │
                         └──────┬──────┘
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌─────────────┐         ┌─────────────┐
             │ SuperPoint  │         │    MiDaS    │
             │  (Features) │         │   (Depth)   │
             └──────┬──────┘         └──────┬──────┘
                    ▼                       ▼
             ┌─────────────┐         ┌─────────────┐
             │  SuperGlue  │         │   YOLOv8    │
             │  (Matching) │         │  (Filter)   │
             └──────┬──────┘         └──────┬──────┘
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │    PnP + RANSAC       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Bundle Adjustment   │
                    └───────────┬───────────┘
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌─────────────┐         ┌─────────────┐
             │   Mapa 3D   │         │   Pose Tt   │
             └──────┬──────┘         └─────────────┘
                    ▼
             ┌─────────────┐         ┌─────────────┐
             │   NetVLAD   │────────▶│ Loop Closing│
             └─────────────┘         └─────────────┘

🚀 Instalação
Pré-requisitos

Sistema Operacional: Ubuntu 20.04/22.04 LTS
GPU: NVIDIA com CUDA 11.8+ (mínimo 6GB VRAM)
Python: 3.8+
RAM: 16GB (recomendado 32GB)

Instalação Rápida
bash# 1. Clonar repositório
git clone https://github.com/rodrigolucas/neural-orbslam.git
cd neural-orbslam

# 2. Criar ambiente virtual
conda create -n neural-orbslam python=3.10
conda activate neural-orbslam

# 3. Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Baixar modelos pré-treinados
python scripts/download_models.py

# 6. Compilar componentes C++ (opcional)
mkdir build && cd build
cmake .. && make -j$(nproc)
Dependências de Sistema
bashsudo apt update
sudo apt install -y \
    build-essential cmake git \
    libopencv-dev libopencv-contrib-dev \
    libeigen3-dev libglew-dev libboost-all-dev \
    libgl1-mesa-glx libegl1-mesa

📖 Uso
Execução Básica
bash# Processar sequência KITTI
python run_slam.py --input data/kitti/00 --output results/

# Com visualização em tempo real
python run_slam.py --input data/kitti/00 --visualize

# Usar apenas câmera monocular
python run_slam.py --input video.mp4 --mode mono
Configuração
yaml# config/default.yaml
model:
  superpoint:
    weights: "models/superpoint_v1.pth"
    nms_radius: 4
    keypoint_threshold: 0.005
    max_keypoints: 1024
  
  midas:
    weights: "models/dpt_large_384.pt"
    input_size: 384
  
  yolov8:
    weights: "models/yolov8n-seg.pt"
    confidence: 0.5
    classes: [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

slam:
  tracking:
    min_matches: 15
    ransac_threshold: 1.0
  
  mapping:
    keyframe_threshold: 0.8
    local_window_size: 10
API Python
pythonfrom neural_orbslam import NeuralORBSLAM

# Inicializar sistema
slam = NeuralORBSLAM(config="config/default.yaml")

# Processar frame
for frame in video_stream:
    pose, map_points = slam.process(frame)
    
    if pose is not None:
        print(f"Posição: {pose.translation}")
        print(f"Pontos no mapa: {len(map_points)}")

# Salvar resultados
slam.save_trajectory("trajectory.txt")
slam.save_map("map.ply")

📊 Avaliação
Executar Benchmarks
bash# Avaliar no KITTI
python evaluate.py --dataset kitti --sequences 00 01 02 03 04 05

# Comparar com baselines
python evaluate.py --dataset kitti --compare orbslam2 orbslam3

# Gerar relatório
python evaluate.py --dataset kitti --report results/report.pdf
Métricas Disponíveis

ATE (Absolute Trajectory Error): Erro absoluto após alinhamento Sim(3)
RPE (Relative Pose Error): Drift relativo entre frames
Taxa de Tracking: Percentual de frames processados com sucesso
FPS: Frames por segundo de processamento


📁 Estrutura do Projeto

neural-orbslam/
├── config/                 # Arquivos de configuração
│   ├── default.yaml
│   └── kitti.yaml
├── data/                   # Datasets e sequências
│   └── kitti/
├── models/                 # Pesos dos modelos pré-treinados
│   ├── superpoint_v1.pth
│   ├── superglue_outdoor.pth
│   ├── dpt_large_384.pt
│   └── yolov8n-seg.pt
├── src/                    # Código fonte
│   ├── neural_orbslam/
│   │   ├── __init__.py
│   │   ├── slam.py
│   │   ├── tracking.py
│   │   ├── mapping.py
│   │   └── loop_closing.py
│   ├── models/
│   │   ├── superpoint.py
│   │   ├── superglue.py
│   │   ├── midas.py
│   │   └── yolov8_filter.py
│   └── utils/
│       ├── geometry.py
│       ├── visualization.py
│       └── evaluation.py
├── scripts/                # Scripts utilitários
│   ├── download_models.py
│   ├── convert_dataset.py
│   └── calibrate_camera.py
├── tests/                  # Testes unitários
├── docs/                   # Documentação
├── results/                # Resultados de experimentos
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md

🔬 Citação

Se você usar este trabalho em sua pesquisa, por favor cite:
bibtex@article{santos2024neuralorbslam,
  title={Neural ORB-SLAM: Uma Extensão Híbrida do ORB-SLAM Integrando 
         Redes Neurais Profundas para SLAM Visual Robusto},
  author={Santos, Rodrigo Lucas},
  journal={Universidade Federal de Ouro Preto},
  year={2024}
}

📚 Referências

ORB-SLAM2 - Mur-Artal & Tardós, 2017
ORB-SLAM3 - Campos et al., 2021
SuperPoint - DeTone et al., 2018
SuperGlue - Sarlin et al., 2020
MiDaS - Ranftl et al., 2021
DROID-SLAM - Teed & Deng, 2021


🤝 Contribuições

Contribuições são bem-vindas! Por favor, leia o CONTRIBUTING.md para detalhes.

Fork o repositório

Crie sua branch (git checkout -b feature/nova-feature)
Commit suas mudanças (git commit -m 'Adiciona nova feature')
Push para a branch (git push origin feature/nova-feature)
Abra um Pull Request


📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

👤 Autor

Rodrigo Lucas Santos

📧 Email: rodrigo.lucas@aluno.ufop.edu.br
🏛️ Instituição: Universidade Federal de Ouro Preto (UFOP)
🔬 Departamento: Departamento de Computação (DECOM)


🙏 Agradecimentos

Departamento de Computação da UFOP pelo suporte computacional
Prof. Dr. Eduardo Luz e Vander Freitas
Comunidade open-source pelos projetos base