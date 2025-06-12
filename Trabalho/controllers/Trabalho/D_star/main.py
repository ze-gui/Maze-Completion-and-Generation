import csv
import math
import time
from typing import Union, List, Dict
import heapq
import itertools

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from controller import Robot, Compass, GPS
import numpy as np

from controllers.Trabalho.metric_graph import VertexInfo, MetricGraph
from controllers.Trabalho.vertex_edge import Vertex, Edge
from controllers.utils import move_robot_to, is_collision_free_line, is_collision_free_point
import networkx as nx


class PriorityQueue:
    # Uma fila de prioridade baseada em heap para armazenar vértices
    def __init__(self):
        self.heap = []                
        self.entry_finder = {}        
        self.REMOVED = object()      
        self.counter = itertools.count()

    def insert(self, u: Vertex, key: tuple):
        if u.id in self.entry_finder:
            self.remove(u)
        count = next(self.counter)
        entry = [key, count, u]
        self.entry_finder[u.id] = entry
        heapq.heappush(self.heap, entry)

    def remove(self, u: Vertex):
        entry = self.entry_finder.pop(u.id)
        entry[2] = self.REMOVED

    def pop(self) -> Vertex:
        while self.heap:
            key, count, u = heapq.heappop(self.heap)
            if u is not self.REMOVED:
                del self.entry_finder[u.id]
                return u
        raise KeyError("pop from an empty priority queue")

    def contains(self, u: Vertex) -> bool:
        return u.id in self.entry_finder


class DStarLite:
    # Implementação do algoritmo D* Lite
    def __init__(self, graph: MetricGraph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.km = 0.0
        self.U = PriorityQueue()
        self.vertex_dict: Dict[int, Vertex] = {v.id: v for v in self.graph.vertex_set}
        self.info_dict: Dict[int, VertexInfo] = {info.id: info for info in self.graph.vertices_info}
        self.last = start

    def h(self, u_id: int) -> float:
        u_info = self.info_dict[u_id]
        goal_info = self.info_dict[self.goal]
        return math.hypot(u_info.x - goal_info.x, u_info.y - goal_info.y)

    def calculate_key(self, u: Vertex) -> tuple:
        k1 = min(u.dist, u.rhs) + self.h(u.id) + self.km
        k2 = min(u.dist, u.rhs)
        return (k1, k2)

    def top_key(self) -> tuple:
        try:
            u = self.U.pop()
        except KeyError:
            return (math.inf, math.inf)
        key = self.calculate_key(u)
        self.U.insert(u, key)
        return key

    def initialize(self):
        for u in self.graph.vertex_set:
            u.dist = math.inf
            u.rhs = math.inf
        goal_v = self.vertex_dict[self.goal]
        goal_v.rhs = 0.0
        self.U.insert(goal_v, self.calculate_key(goal_v))

    def update_vertex(self, u: Vertex):
        if u.id != self.goal:
            u.rhs = min(e.weight + e.dest.dist for e in u.adj)
        if self.U.contains(u):
            self.U.remove(u)
        if u.dist != u.rhs:
            self.U.insert(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while (self.top_key() < self.calculate_key(self.vertex_dict[self.start]) or
               self.vertex_dict[self.start].rhs != self.vertex_dict[self.start].dist):
            u = self.U.pop()
            if u.dist > u.rhs:
                u.dist = u.rhs
                for pred in self.get_predecessors(u.id):
                    self.update_vertex(pred)
            else:
                u.dist = math.inf
                self.update_vertex(u)
                for pred in self.get_predecessors(u.id):
                    self.update_vertex(pred)

    def get_predecessors(self, u_id: int) -> List[Vertex]:
        preds = []
        for v in self.graph.vertex_set:
            for e in v.adj:
                if e.dest.id == u_id:
                    preds.append(v)
                    break
        return preds

    def update_edge(self, u_id: int, v_id: int, new_weight: float):
        u = self.vertex_dict[u_id]
        v = self.vertex_dict[v_id]
        for e in u.adj:
            if e.dest.id == v_id:
                e.weight = new_weight
                break
        for e in v.adj:
            if e.dest.id == u_id:
                e.weight = new_weight
                break
        self.km += self.h(self.last)
        self.last = self.start
        self.update_vertex(u)
        self.update_vertex(v)
        self.compute_shortest_path()

    def get_next(self, u_id: int) -> Union[int, None]:
        u = self.vertex_dict[u_id]
        best, best_val = None, math.inf
        for e in u.adj:
            val = e.weight + e.dest.dist
            if val < best_val:
                best_val, best = val, e.dest.id
        return best

    def get_path(self) -> List[int]:
        path, u = [], self.start
        while u != self.goal:
            path.append(u)
            u = self.get_next(u)
        path.append(self.goal)
        return path


def find_closest_accessible_vertex(x: float, y: float, vertices_info: List[VertexInfo], obstacle_cloud: np.ndarray) -> Union[VertexInfo, None]:
    # Encontrar o vértice mais próximo que é acessível a partir do ponto (x, y)
    closest_accessible_vertex = None
    closest_distance = math.inf
    for vertex_info in vertices_info:
        distance = math.hypot(x - vertex_info.x, y - vertex_info.y)
        if distance >= closest_distance:
            continue
        if is_collision_free_line(x, y, vertex_info.x, vertex_info.y, obstacle_cloud):
            closest_accessible_vertex = vertex_info
            closest_distance = distance
    return closest_accessible_vertex


def create_grid_graph(initial_pos: (float, float), final_pos: (float, float), obstacle_cloud: np.ndarray) -> MetricGraph:
    # Criar um grafo métrico com uma grade de pontos
    grid_graph = MetricGraph()

    # Definir os parâmetros
    n_xy_divisions = 32
    max_x = 2.40
    max_y = 2.40
    # para labirintos maiores que 10
    # n_xy_divisions = 64
    # max_x = 4.80
    # max_y = 4.80
    min_x_offset = -0.2
    min_y_offset = -0.2
    x_increment = max_x / n_xy_divisions
    y_increment = max_y / n_xy_divisions
    cur_index = 0
    for i in range(n_xy_divisions):
        x = min_x_offset + i * x_increment
        for j in range(n_xy_divisions):
            y = min_y_offset + j * y_increment
            grid_graph.add_vertex(cur_index, (x, y), 'blue')
            cur_index += 1

    # Adicionar os pontos inicial e final como vértices adicionais
    start_idx = cur_index
    grid_graph.add_vertex(start_idx, initial_pos, 'green')
    goal_idx = cur_index + 1
    grid_graph.add_vertex(goal_idx, final_pos, 'green')

    # Conectar o ponto inicial ao ponto mais próximo na grade usando arestas
    s_vi = find_closest_accessible_vertex(initial_pos[0], initial_pos[1], grid_graph.vertices_info[:-2], obstacle_cloud)
    if s_vi is None:
        print("Initial position not accessible.")
        return grid_graph
    dist_s = math.hypot(initial_pos[0] - s_vi.x, initial_pos[1] - s_vi.y)
    grid_graph.add_edge(start_idx, s_vi.id, dist_s)
    grid_graph.add_edge(s_vi.id, start_idx, dist_s)

    # Conectar o ponto final ao ponto mais próximo na grade usando arestas
    f_vi = find_closest_accessible_vertex(final_pos[0], final_pos[1], grid_graph.vertices_info[:-2], obstacle_cloud)
    if f_vi is None:
        print("Final position not accessible.")
        return grid_graph
    dist_f = math.hypot(final_pos[0] - f_vi.x, final_pos[1] - f_vi.y)
    grid_graph.add_edge(goal_idx, f_vi.id, dist_f)
    grid_graph.add_edge(f_vi.id, goal_idx, dist_f)

    # Adicionar arestas entre os vértices adjacentes na grade
    for idx in range(n_xy_divisions * n_xy_divisions):
        i = idx // n_xy_divisions
        j = idx % n_xy_divisions
        u = grid_graph.vertices_info[idx]
        if i > 0:
            v = grid_graph.vertices_info[idx - n_xy_divisions]
            if is_collision_free_line(u.x, u.y, v.x, v.y, obstacle_cloud):
                grid_graph.add_edge(idx, idx - n_xy_divisions, x_increment)
                grid_graph.add_edge(idx - n_xy_divisions, idx, x_increment)
        if j > 0:
            v = grid_graph.vertices_info[idx - 1]
            if is_collision_free_line(u.x, u.y, v.x, v.y, obstacle_cloud):
                grid_graph.add_edge(idx, idx - 1, y_increment)
                grid_graph.add_edge(idx - 1, idx, y_increment)

    return grid_graph


def main() -> None:
    robot = Robot()

    custom_maps = '../../../worlds/custom_maps/'
    map_name = 'maze_3_1'
    obs_file = custom_maps + map_name + '_points.csv'
    final_position = (1.1, 2.2)
    # para labirintos maiores que 10
    # final_position = (2.2, 4.4)

    timestep = int(robot.getBasicTimeStep())

    compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps = robot.getDevice('gps')
    gps.enable(timestep)
    robot.step()

    # Ler as coordenadas do robô
    gps_read = gps.getValues()
    robot_position = (gps_read[0], gps_read[1])

    # Ler as coordenadas dos obstáculos
    obs_points = []
    with open(obs_file, 'r') as f:
        for row in csv.reader(f):
            obs_points.append([float(row[0]), float(row[1]), 0.0])
    obstacle_cloud = np.asarray(obs_points)

    # Verificar se a posição final do robô é livre de colisões
    if not is_collision_free_point(final_position[0], final_position[1], obstacle_cloud):
        print("Final position collides.")
        return

    # Criar o grafo e executar o D* Lite
    t0 = time.time()
    grid_graph = create_grid_graph(robot_position, final_position, obstacle_cloud)
    print(f"Elapsed time for creating the grid graph :  {time.time() - t0}  seconds")
    start_id = len(grid_graph.vertex_set) - 2
    goal_id  = len(grid_graph.vertex_set) - 1
    planner = DStarLite(grid_graph, start_id, goal_id)
    planner.initialize()
    t1 = time.time()
    planner.compute_shortest_path()
    print(f"Elapsed time for D* :  {time.time() - t1}  seconds")

    # Executar o caminho dinâmico
    path_ids = [start_id]
    current_id = start_id
    steps = 0
    t2 = time.time()
    while current_id != goal_id:
        next_id = planner.get_next(current_id)
        if next_id is None:
            print("No path!  Stuck at vertex", current_id)
            break
        robot.step()
        steps += 1
        gps_read = gps.getValues()
        robot_position = (gps_read[0], gps_read[1])
        comp = compass.getValues()
        robot_orientation = math.atan2(comp[0], comp[1])
        move_robot_to(robot,
                      robot_position,
                      robot_orientation,
                      (grid_graph.vertices_info[next_id].x,
                       grid_graph.vertices_info[next_id].y),
                      0.1, math.pi)
        current_id = next_id
        planner.start = current_id
        planner.km += planner.h(planner.last)
        planner.last = planner.start
        path_ids.append(current_id)
        for e in list(grid_graph.vertex_set[current_id].adj):
            u_info = grid_graph.vertices_info[current_id]
            v_info = grid_graph.vertices_info[e.dest.id]
            if not is_collision_free_line(u_info.x, u_info.y,
                                          v_info.x, v_info.y,
                                          obstacle_cloud):
                planner.update_edge(current_id, e.dest.id, math.inf)

    print(f"Elapsed time for path completion :  {time.time() - t2}  seconds")
    print(f"To completion the robot needed :  {steps}  steps")

    # Mudar as cores dos vértices do caminho para verde e os outros para vermelho
    new_colors = {vid: 'green' for vid in path_ids}
    for v in grid_graph.vertex_set:
        if v.dist == math.inf:
            new_colors[v.id] = 'red'
    nx.set_node_attributes(grid_graph.visual_graph, new_colors, 'color')

    # Mostar o grafo métrico com os obstáculos e o caminho encontrado
    fig, ax = plt.subplots()
    pos = {info.id: (info.x, info.y) for info in grid_graph.vertices_info}
    nx.draw_networkx(
        grid_graph.visual_graph, pos,
        node_size=10,
        node_color=[new_colors.get(n, 'blue')
                    for n in grid_graph.visual_graph.nodes()],
        with_labels=False
    )
    pat = [Rectangle((pt[0], pt[1]), 0.001, 0.001) for pt in obstacle_cloud]
    col = PatchCollection(pat)
    col.set_edgecolor('black')
    col.set_linewidth(1)
    ax.add_collection(col)
    plt.show()


if __name__ == '__main__':
    main()
