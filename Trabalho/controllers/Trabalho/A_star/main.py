import csv
import math
import time
from typing import Union, Dict

from matplotlib import pyplot as plt, patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from controller import Robot, Compass, GPS
import numpy as np

from controllers.Trabalho.metric_graph import VertexInfo, MetricGraph
from controllers.Trabalho.vertex_edge import Vertex
from controllers.utils import move_robot_to, is_collision_free_line, is_collision_free_point
import networkx as nx


def find_closest_accessible_vertex(x: float, y: float, vertices_info: [VertexInfo], obstacle_cloud: np.ndarray) -> Union[VertexInfo, None]:
    # Encontrar o vértice mais próximo que é acessível a partir do ponto (x, y)
    closest_accessible_vertex: Union[VertexInfo, None] = None
    closest_distance: float = math.inf
    for vertex_info in vertices_info:
        distance: float = math.hypot(x - vertex_info.x, y - vertex_info.y)
        if distance >= closest_distance:
            continue
        if is_collision_free_line(x, y, vertex_info.x, vertex_info.y, obstacle_cloud):
            closest_accessible_vertex = vertex_info
            closest_distance = distance
    return closest_accessible_vertex


def create_grid_graph(initial_pos: (float, float), final_pos: (float, float), obstacle_cloud: np.ndarray) -> MetricGraph:
    # Criar um grafo métrico com um grafo de pontos
    grid_graph = MetricGraph()

    # Definir os parâmetros
    n_xy_divisions: int = 32
    max_x: float = 2.40
    max_y: float = 2.40
    # para labirintos maiores que 10
    # n_xy_divisions: int = 64
    # max_x: float = 4.80
    # max_y: float = 4.80
    min_x_offset: float = -0.2
    x_increment: float = max_x / n_xy_divisions
    min_y_offset: float = -0.2
    y_increment: float = max_y / n_xy_divisions
    cur_index: int = 0
    for i in range(n_xy_divisions):
        x: float = min_x_offset + i * x_increment
        for j in range(n_xy_divisions):
            y: float = min_y_offset + j * y_increment
            # Add a vertex to point (x,y)
            grid_graph.add_vertex(cur_index, (x, y), 'blue')
            cur_index += 1

    # Adicionar os pontos inicial e final como vértices adicionais
    additional_points: [(float, float)] = [initial_pos, final_pos]
    for point in additional_points:
        grid_graph.add_vertex(cur_index, point, 'green')
        cur_index += 1

    # Conectar o ponto inicial ao ponto mais próximo na grade usando arestas
    closest_vertex_to_initial: Union[VertexInfo, None] = find_closest_accessible_vertex(initial_pos[0], initial_pos[1], grid_graph.vertices_info[:-2], obstacle_cloud)
    if closest_vertex_to_initial is None:
        print("Initial position ", initial_pos, " is not accessible by any point in the grid.")
        return grid_graph
    grid_graph.add_edge(len(grid_graph.vertices_info) - 2, closest_vertex_to_initial.id,
                        math.hypot(initial_pos[0] - closest_vertex_to_initial.x, initial_pos[1] - closest_vertex_to_initial.y))

    # Conectar o ponto final ao ponto mais próximo na grade usando arestas
    closest_vertex_to_final: Union[VertexInfo, None] = find_closest_accessible_vertex(final_pos[0], final_pos[1], grid_graph.vertices_info[:-2], obstacle_cloud)
    if closest_vertex_to_final is None:
        print("Final position ", final_pos, " is not accessible by any point in the grid.")
        return grid_graph
    grid_graph.add_edge(closest_vertex_to_final.id, len(grid_graph.vertices_info) - 1,
                        math.hypot(final_pos[0] - closest_vertex_to_final.x,
                                   final_pos[1] - closest_vertex_to_final.y))

    # Adicionar arestas entre os vértices adjacentes na grade
    cur_index = 0
    for i in range(n_xy_divisions):
        for j in range(n_xy_divisions):
            if i > 0 and is_collision_free_line(grid_graph.vertices_info[cur_index].x, grid_graph.vertices_info[cur_index].y, grid_graph.vertices_info[cur_index - n_xy_divisions].x, grid_graph.vertices_info[cur_index - n_xy_divisions].y, obstacle_cloud):
                grid_graph.add_edge(cur_index, cur_index - n_xy_divisions, x_increment)
                grid_graph.add_edge(cur_index - n_xy_divisions, cur_index, x_increment)
            if j > 0 and is_collision_free_line(grid_graph.vertices_info[cur_index].x, grid_graph.vertices_info[cur_index].y, grid_graph.vertices_info[cur_index - 1].x, grid_graph.vertices_info[cur_index - 1].y, obstacle_cloud):
                grid_graph.add_edge(cur_index, cur_index - 1, y_increment)
                grid_graph.add_edge(cur_index - 1, cur_index, y_increment)
            cur_index += 1

    return grid_graph


def main() -> None:
    robot: Robot = Robot()

    custom_maps_filepath: str = '../../../worlds/custom_maps/'
    map_name: str = 'maze_3_1'
    obstacle_points_filename: str = custom_maps_filepath + map_name + '_points.csv'
    final_position: (float, float) = (1.1, 2.2)
    # para labirintos maiores que 10
    # final_position: (float, float) = (2.2, 4.4)

    timestep: int = int(robot.getBasicTimeStep())  # in ms

    compass: Compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = robot.getDevice('gps')
    gps.enable(timestep)
    robot.step()

    # Ler as coordenadas do robô
    gps_readings: [float] = gps.getValues()
    robot_position: (float, float) = (gps_readings[0], gps_readings[1])

    # Ler as coordenadas dos obstáculos
    obstacle_points: [(float, float, float)] = []
    with open(obstacle_points_filename, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            obstacle_points.append([float(row[0]), float(row[1]), 0.0])
    obstacle_cloud: np.ndarray = np.asarray(obstacle_points)

    # Verificar se a posição inicial do robô é livre de colisões
    if not is_collision_free_point(final_position[0], final_position[1], obstacle_cloud):
        print("Final position ", final_position, " is colliding with an obstacle")
        return

    # Criar o grafo e executar o A*
    start = time.time()
    grid_graph = create_grid_graph(robot_position, final_position, obstacle_cloud)
    end = time.time()
    print("Elapsed time for creating the grid graph : ", end - start, " seconds")
    vertex_positions: Dict[int, (float, float)] = nx.get_node_attributes(grid_graph.visual_graph, 'pos')
    vertex_distances: Dict[int, float] = {}
    for (id, position) in vertex_positions.items():
        vertex_distances[id] = math.hypot(position[0] - final_position[0],
                                          position[1] - final_position[1])

    def dist_func(v: Vertex):
        return vertex_distances[v.id]
    start = time.time()
    grid_graph.a_star(len(grid_graph.vertex_set) - 2, dist_func, len(grid_graph.vertex_set) - 1)
    end = time.time()
    print("Elapsed time for A* : ", end - start, " seconds")
    path: [Vertex] = grid_graph.get_path(len(grid_graph.vertex_set) - 2, len(grid_graph.vertex_set) - 1)

    # Mudar as cores dos vértices do caminho para verde e os outros para vermelho
    new_vertex_colors: dict = {}
    for vertex in path:
        new_vertex_colors[vertex.id] = "green"
    for vertex in grid_graph.vertex_set:
        if vertex.dist == math.inf:
            new_vertex_colors[vertex.id] = "red"
    nx.set_node_attributes(grid_graph.visual_graph, new_vertex_colors, 'color')
    pat: [patches.Rectangle] = []
    for point in obstacle_cloud:
        pat.append(Rectangle((point[0], point[1]), 0.001, 0.001,
                             linewidth=1, edgecolor='black', facecolor='none'))

    # Mostar o grafo métrico com os obstáculos e o caminho encontrado
    fig, ax = plt.subplots()
    nx.draw_networkx(grid_graph.visual_graph, vertex_positions, node_size=10,
                     node_color=nx.get_node_attributes(grid_graph.visual_graph, 'color').values(),
                     with_labels=False)
    col: PatchCollection = PatchCollection(pat)
    col.set_edgecolor('black')
    col.set_linewidth(1)
    ax.add_collection(col)
    plt.show()

    # Mover o robô ao longo do caminho encontrado
    start = time.time()
    steps = -1
    for vertex in path:
        robot.step()
        steps += 1
        gps_readings: [float] = gps.getValues()
        robot_position: (float, float) = (gps_readings[0], gps_readings[1])
        compass_readings: [float] = compass.getValues()
        robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])

        move_robot_to(robot, robot_position, robot_orientation, vertex_positions[vertex.id], 0.1, math.pi)
    end = time.time()
    print("Elapsed time for path completion : ", end - start, " seconds")
    print("To completion the robot needed : ", steps, " steps")


if __name__ == '__main__':
    main()
