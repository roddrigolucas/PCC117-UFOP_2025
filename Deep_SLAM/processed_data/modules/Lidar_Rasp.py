## O código realiza as seguintes tarefas:

## Coleta de dados LIDAR com o RPLidar.

## Normalização dos dados para uma área de 4x6 metros.

## Plotagem progressiva do SLAM, com a trilha do robô sendo desenhada em tempo real.

## Atualizações do mapa a cada etapa, mostrando como o robô vai descobrindo o ambiente.

import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar
import time

# Função para normalizar os pontos para 4 metros de largura e 6 metros de altura
def normalize_point(point, original_width=8, original_height=12.5, new_width=4, new_height=6):
    x, y = point
    x_norm = (x / original_width) * new_width
    y_norm = (y / original_height) * new_height
    return x_norm, y_norm

# Função para converter coordenadas polares para cartesianas
def polar_to_cartesian(angle, distance):
    rad = np.deg2rad(angle)
    x = distance * np.cos(rad)
    y = distance * np.sin(rad)
    return x, y

# Função para simular o SLAM progressivo com RPLidar
def run_slam():
    lidar = RPLidar('/dev/ttyUSB0')  # Alterar conforme a porta do seu dispositivo

    try:
        # Preparar o gráfico
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.set_title('SLAM Progressivo com RPLidar')
        ax.set_xlabel('Eixo X (m)')
        ax.set_ylabel('Eixo Y (m)')

        # Inicializar listas de pontos
        all_points_x = []
        all_points_y = []
        trajeto_robo = []

        # Leitura dos dados do LIDAR
        etapas = 6  # Definindo o número de etapas do SLAM
        leitura_por_etapa = 100  # Número de leituras por etapa
        leitura_total = 0

        # Iniciar o loop de leitura do LIDAR
        for i, scan in enumerate(lidar.iter_scans()):
            # Processa a leitura do LIDAR (qualidade, ângulo, distância)
            points = []
            for quality, angle, distance in scan:
                if distance > 0:
                    x, y = polar_to_cartesian(angle, distance / 1000.0)  # Convertendo mm para metros
                    x, y = normalize_point((x, y))  # Normalizando para 4m x 6m
                    points.append((x, y))
            
            # Adiciona os pontos ao conjunto total de pontos
            all_points_x.extend([p[0] for p in points])
            all_points_y.extend([p[1] for p in points])
            
            # Adiciona a posição do robô ao trajeto
            trajeto_robo.append((points[-1][0], points[-1][1]))  # Último ponto de cada varredura

            leitura_total += len(points)
            
            # Atualiza o gráfico a cada etapa
            if leitura_total >= leitura_por_etapa * (i + 1):
                ax.clear()
                ax.scatter(all_points_x, all_points_y, s=2, c='black', alpha=0.75)  # Pontos LIDAR
                
                # Desenha a trilha do robô
                trajeto_x, trajeto_y = zip(*trajeto_robo)
                ax.plot(trajeto_x, trajeto_y, color='red', linewidth=1.5, label='Caminho do robô')

                ax.set_xlim(0, 4)
                ax.set_ylim(0, 6)
                ax.set_aspect('equal')
                ax.set_title(f'SLAM - Etapa {i+ 1}')
                ax.set_xlabel('Eixo X (m)')
                ax.set_ylabel('Eixo Y (m)')
                ax.legend(loc='lower left')

                plt.pause(0.1)
            
            # Termina após atingir 6 etapas
            if i >= etapas - 1:
                break

        plt.ioff()  # Desativa o modo interativo
        plt.show()
    finally:
        lidar.stop()
        lidar.disconnect()

# Executa o SLAM
if __name__ == "__main__":
    run_slam()

pip install --upgrade rplidar