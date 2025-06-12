import csv
import math
import time
import random
from typing import Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from controller import Robot, Compass, GPS
from controllers.Trabalho.metric_graph import MetricGraph
from controllers.utils import is_collision_free_point, is_collision_free_line, move_robot_to


def collision_free_segment(x1: float, y1: float, x2: float, y2: float, obstacle_tree: cKDTree, obstacle_cloud: np.ndarray, padding: float) -> bool:
    # Verifica se o segmento de linha entre (x1, y1) e (x2, y2) está livre de colisões
    midx = (x1 + x2) / 2.0
    midy = (y1 + y2) / 2.0
    half_len = math.hypot(x2 - x1, y2 - y1) / 2.0
    radius = half_len + padding

    idxs = obstacle_tree.query_ball_point((midx, midy), radius)
    local_obs = obstacle_cloud[idxs] if idxs else np.empty((0, 3))
    return is_collision_free_line(x1, y1, x2, y2, local_obs)


def create_rrt(initial_position: Tuple[float, float], final_position: Tuple[float, float],obstacle_cloud: np.ndarray) -> Tuple[bool, MetricGraph]:
    # Implementação do algoritmo RRT (Rapidly-exploring Random Tree)
    # Definir os paremetros
    goal_bias = 0.05      
    max_iter = 100_000   
    step_size = 0.2      
    rebuild_kdtree_every = 200       
    max_x = 2.4
    max_y = 2.4
    # para labirintos maiores que 10
    #max_x = 4.8
    #max_y = 4.8

    # Construir a árvore KD para os obstáculos
    obs_pts_2d = obstacle_cloud[:, :2]
    obstacle_tree = cKDTree(obs_pts_2d)

    # Inicializar o grafo RRT e adicionar o vértice inicial
    rrt_graph = MetricGraph()
    rrt_graph.add_vertex(0, initial_position, 'blue')
    vertex_coords = [initial_position]
    vertex_tree = cKDTree(vertex_coords)
    cur_index = 1

    # Verificar se o segmento inicial-final está livre de colisões
    if collision_free_segment(initial_position[0], initial_position[1], final_position[0], final_position[1], obstacle_tree, obstacle_cloud, padding=step_size):
        dist = math.hypot(final_position[0] - initial_position[0], final_position[1] - initial_position[1])
        rrt_graph.add_vertex(cur_index, final_position, 'blue')
        rrt_graph.add_edge(0, cur_index, dist, True)
        return True, rrt_graph

    # Começar o loop RRT
    for k in range(max_iter):
        if k % 10_000 == 0:
            print(f"[RRT] iter={k}/{max_iter}, verts={len(vertex_coords)}")

        # caminhamento aleatório com viés para o objetivo
        if random.random() < goal_bias:
            rx, ry = final_position
        else:
            rx, ry = random.uniform(0, max_x), random.uniform(0, max_y)
            if not is_collision_free_point(rx, ry, obstacle_cloud):
                continue

        # Encontrar o vértice mais próximo
        dist, idx = vertex_tree.query((rx, ry))
        vx, vy= vertex_coords[idx]
        step = min(step_size, dist)
        nxp = vx + step * (rx - vx) / dist
        nyp = vy + step * (ry - vy) / dist

        # Verificar se o segmento de linha está livre de colisões
        if not collision_free_segment(vx, vy, nxp, nyp, obstacle_tree, obstacle_cloud, padding=0.05):
            continue

        # Adicionar o novo vértice ao grafo RRT
        rrt_graph.add_vertex(cur_index, (nxp, nyp), 'blue')
        rrt_graph.add_edge(idx, cur_index, step, True)
        vertex_coords.append((nxp, nyp))
        cur_index += 1

        # Reconstruir a árvore KD a cada rebuild_kdtree_every iterações
        if len(vertex_coords) % rebuild_kdtree_every == 0:
            vertex_tree = cKDTree(vertex_coords)

        # Verificar se o novo vértice está próximo do objetivo
        if collision_free_segment(nxp, nyp, final_position[0], final_position[1],obstacle_tree, obstacle_cloud,padding=0.05):
            last_dist = math.hypot(final_position[0] - nxp, final_position[1] - nyp)
            rrt_graph.add_vertex(cur_index, final_position, 'blue')
            rrt_graph.add_edge(cur_index - 1, cur_index, last_dist, True)
            return True, rrt_graph

    return False, rrt_graph


def main() -> None:
    robot   = Robot()

    custom_maps = '../../../worlds/custom_maps/'
    map_name = 'maze_3_1'
    obs_file = custom_maps + map_name + '_points.csv'
    final_pos = (1.1, 2.2)
    # para labirintos maiores que 10
    # final_pos = (2.2, 4.4)

    timestep = int(robot.getBasicTimeStep())

    compass: Compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = robot.getDevice('gps')
    gps.enable(timestep)
    robot.step()

    # Ler as coordenadas dos obstáculos
    with open(obs_file, 'r') as f:
        reader = csv.reader(f)
        obs = np.array([[float(x), float(y), 0.0] for x, y in reader])

    # Verificar se a posição final do robô é livre de colisões
    if not is_collision_free_point(final_pos[0], final_pos[1], obs):
        print("Final position colliding with obstacle:", final_pos)
        return

    # Ler as coordenadas do rob
    gx, gy, _ = gps.getValues()
    start_pos = (gx, gy)

    # Criar o grafo e executar o RRT
    t0 = time.time()
    found, rrt = create_rrt(start_pos, final_pos, obs)
    t1 = time.time()
    print(f"RRT build time: {t1-t0:}s")
    if not found:
        print("No path found.")
        return
    path = rrt.get_path(0, len(rrt.vertex_set)-1)

    # Mudar as cores dos vértices do caminho para verde e os outros para vermelho
    pos = nx.get_node_attributes(rrt.visual_graph, 'pos')
    colors = {v.id: 'green' for v in path}
    nx.set_node_attributes(rrt.visual_graph, colors, 'color')

    # Mostar o grafo métrico com os obstáculos e o caminho encontrado
    fig, ax = plt.subplots()
    patches = [Rectangle((x, y), 0.001, 0.001) for x, y, _ in obs]
    pc = PatchCollection(patches, edgecolor='black', facecolor='none', linewidth=1)
    ax.add_collection(pc)

    nx.draw_networkx(rrt.visual_graph, pos, node_size=10, node_color=list(nx.get_node_attributes(rrt.visual_graph, 'color').values()), with_labels=False, ax=ax)
    plt.show()

    # Mover o robô ao longo do caminho encontrado
    t2 = time.time()
    steps = 0
    for v in path:
        robot.step()
        steps += 1
        gx, gy, _ = gps.getValues()
        ori = math.atan2(*compass.getValues()[:2])
        move_robot_to(robot, (gx, gy), ori, pos[v.id], 0.1, math.pi)
    t3 = time.time()
    print(f"Path execution: {t3-t2:}s in {steps} steps")


if __name__ == '__main__':
    main()
