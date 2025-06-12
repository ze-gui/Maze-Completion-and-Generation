"""
A simple implementation of a graph, including a relax operation for single source shortest-path algorithms.
By: Gonçalo Leão
"""
from __future__ import annotations
import math
from typing import Union, Callable

from controllers.Trabalho.vertex_edge import Vertex, Edge
from controllers.Trabalho.mutable_priority_queue import MutablePriorityQueue


class Graph:
    def __init__(self):
        self.vertex_set: [Vertex] = []

    # Finds a vertex with a given ID.
    def find_vertex(self, id: int) -> Union[Vertex, None]:
        for v in self.vertex_set:
            if v.id == id:
                return v
        return None

    # Adds a vertex with a given content or info to a graph.
    # Returns true if successful, and false if a vertex with that content already exists.
    def add_vertex(self, id: int) -> bool:
        if self.find_vertex(id) is not None:
            return False
        self.vertex_set.append(Vertex(id))
        return True

    # Adds an edge to the graph, given the contents of the origin and
    # destination vertices and the edge weight(w).
    # Returns true if successful, and false if the source or destination vertex does not exist.
    def add_edge(self, origin: int, dest: int, weight: float, set_as_path: bool = False) -> bool:
        v1: Vertex = self.find_vertex(origin)
        v2: Vertex = self.find_vertex(dest)
        if v1 is None or v2 is None:
            return False
        new_edge: Edge = v1.add_edge(v2, weight)
        if set_as_path:
            v2.path = new_edge
        return True

    def merge(self, other: Graph) -> None:
        self.vertex_set.extend(other.vertex_set)

    def dijkstra(self, origin: int, dest: int = -1) -> bool:
        return self.a_star(origin, lambda _: 0, dest)

    # Arguments:
    # origin: Source node ID
    # heuristic_cost_function: function that estimates the distance from the vertex argument to the goal
    # dest: Destination node ID (-1 to search for every accessible node)
    def a_star(self, origin: int, heuristic_cost_function: Callable[[Vertex], float], dest: int = -1) -> bool:
        # Initialize the vertices
        for v in self.vertex_set:
            v.dist = math.inf
            v.path = None

        # Retrieve initial vertex s and set its dist to 0
        s: Vertex = self.find_vertex(origin)
        s.dist = 0

        # Create the priority queue and add s
        q: MutablePriorityQueue = MutablePriorityQueue()
        q.insert(s)

        # Process the vertices
        while not q.empty():
            v: Vertex = q.extract_min()
            if v.id == dest:  # the destination was found so the search can stop
                return True
            for e in v.adj:
                old_dist: float = e.dest.dist
                if self.relax(e, heuristic_cost_function):  # a shorter path was found
                    if old_dist == math.inf:  # new vertex was found
                        q.insert(e.dest)
                    else:  # a shorter path to an unprocessed vertex was found (unprocessed = vertex still in the queue)
                        q.decrease_key(e.dest)
        return False

    # Analyzes an edge in single source shortest path algorithm.
    # Returns true if the target vertex was relaxed (dist, path).
    # Used by all single-source shortest path algorithms.
    def relax(self, edge: Edge, heuristic_cost_function: Callable[[Vertex], float]) -> bool:
        if edge.origin.dist + edge.weight < edge.dest.dist:
            # Update edge.dest.dist, edge.dest.path and edge.dest.cost
            edge.dest.dist = edge.origin.dist + edge.weight
            edge.dest.path = edge
            edge.dest.cost = edge.dest.dist + heuristic_cost_function(edge.dest)
            return True
        else:
            return False

    def get_path(self, origin: int, dest: int) -> [Vertex]:
        path: [Vertex] = []

        # Add the destination vertex to the path
        v: Vertex = self.find_vertex(dest)
        if v is None or v.dist == math.inf:  # missing or disconnected vertex
            return []
        path.append(v)

        # Follow the path in reverse order and add each vertex to the path
        while v.path is not None:
            v = v.path.origin
            path.append(v)
        # Reverse the path
        path.reverse()

        # Check that the origin vertex was found, if not return []
        if len(path) == 0 or path[0].id != origin:
            return []
        return path
