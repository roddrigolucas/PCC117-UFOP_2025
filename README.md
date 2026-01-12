neural-orbslam/
├── config/                  # Arquivos de configuração
│   ├── default.yaml
│   └── kitti.yaml
│
├── data/                    # Datasets e sequências
│   └── kitti/
│
├── checkpoints/             # Pesos dos modelos pré-treinados (renomeado de 'models')
│   ├── superpoint_v1.pth
│   ├── superglue_outdoor.pth
│   ├── dpt_large_384.pt
│   └── yolov8n-seg.pt
│
├── src/                     # Código fonte principal
│   ├── neural_orbslam/      # Núcleo do SLAM
│   │   ├── __init__.py
│   │   ├── slam.py
│   │   ├── tracking.py
│   │   ├── mapping.py
│   │   └── loop_closing.py
│   │
│   ├── models/              # Implementações dos modelos
│   │   ├── superpoint.py
│   │   ├── superglue.py
│   │   ├── midas.py
│   │   └── yolov8_filter.py
│   │
│   └── utils/               # Funções auxiliares
│       ├── geometry.py
│       ├── visualization.py
│       └── evaluation.py
│
├── scripts/                 # Scripts utilitários
│   ├── download_models.py
│   ├── convert_dataset.py
│   └── calibrate_camera.py
│
├── tests/                   # Testes unitários
│
├── docs/                    # Documentação
│
├── results/                 # Resultados de experimentos
│
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
