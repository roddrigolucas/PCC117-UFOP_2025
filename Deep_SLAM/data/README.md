3. Dependências do ORB-SLAM Base
bash# Bibliotecas essenciais
sudo apt install build-essential cmake git
sudo apt install libopencv-dev libopencv-contrib-dev
sudo apt install libeigen3-dev
sudo apt install libglew-dev libboost-all-dev

# Pangolin (visualização 3D)
git clone https://github.com/stevenlovegrove/Pangolin.git

# g2o (otimização de grafos)
git clone https://github.com/RainerKuemmerle/g2o.git

# DBoW2 (vocabulário visual)
git clone https://github.com/dorian3d/DBoW2.git

4. Modelos de Deep Learning Pré-treinados
MóduloModeloTamanhoFonteSuperPointsuperpoint_v1.pth~5MBMagicLeapSuperGluesuperglue_outdoor.pth~12MBMagicLeapMiDaSdpt_large_384.pt~1.3GBIntel ISLYOLOv8yolov8n-seg.pt~7MBUltralyticsNetVLADpittsburgh_trained.pth~100MBNanne

5. Bibliotecas Python
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python opencv-contrib-python
pip install numpy scipy matplotlib
pip install timm                    # Para MiDaS/DPT
pip install ultralytics             # Para YOLOv8
pip install kornia                  # Geometria diferenciável
pip install open3d                  # Visualização de nuvens de pontos
pip install evo                     # Avaliação de trajetórias

6. Sensores
SensorObrigatórioEspecificação MínimaCâmera RGB✅ Sim640×480 @ 30fpsCâmera Estéreo❌ OpcionalBaseline > 10cmIMU❌ Opcional6-DOF (accel + gyro)LiDAR❌ Opcional2D/3D scanning

7. Datasets para Teste/Calibração
bash# KITTI Odometry (principal benchmark)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip

# TUM RGB-D (indoor)
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz

# Calibração de câmera
pip install opencv-python
# Usar padrão checkerboard para calibração intrínseca


# 1. Clonar repositório
git clone https://github.com/rodrigolucas/neural-orbslam.git
cd neural-orbslam

# 2. Criar ambiente virtual
conda create -n neural-orbslam python=3.10
conda activate neural-orbslam

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar modelos pré-treinados
python scripts/download_models.py

# 5. Compilar componentes C++
mkdir build && cd build
cmake .. && make -j$(nproc)

# 6. Testar
python run_slam.py --input data/kitti/00 --output results/
