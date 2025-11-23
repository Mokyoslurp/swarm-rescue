from typing import Optional

from swarm_rescue.solutions.my_solution.mapping.vertex import Vertex


class GraphADT:
    """A Graph structure"""

    def __init__(self):
        self._edges: dict[Vertex, dict[Vertex, float]] = {}
        self._vertices: dict[Vertex, bool] = {}

    def append(self, vertex_from: Vertex, vertex_to: Vertex, weight: float):
        """Adds a new weighted edge between two vertices. Adds the vertices if not already
            in the graph.

        Args:
            vertex_from (Vertex): the first vertex
            vertex_to (Vertex): the second vertex
            weight (float): the weight of the edge
        """

        if vertex_from not in self._vertices:
            self._vertices[vertex_from] = False
            self._edges[vertex_from] = {}

        if vertex_to not in self._vertices:
            self._vertices[vertex_to] = False
            self._edges[vertex_to] = {}

        self._edges[vertex_from][vertex_to] = weight
        self._edges[vertex_to][vertex_from] = weight

    @property
    def edges(self):
        """Gets edges of the graph

        Returns:
            dict[Vertex, dict[Vertex, float]]: Edges dictionnary containing vertices as index,
            and a second dictionnary as value, containing itself the second vertex of
            the edge as the key and the weight of the edge as the value
        """
        return self._edges

    @property
    def vertices(self):
        """Return all vertices

        Returns:
            list[Vertex]: list of the vertices
        """
        return list(self._vertices.keys())

    def edge(self, vertex: Vertex) -> list[Vertex]:
        """Returns all edges of a vertex

        Args:
            vertex (Vertex): the vertex

        Returns:
            list[Vertex]: list of vertices linked by edges
        """
        if vertex in self._edges:
            return list(self._edges[vertex].keys())
        return []

    def edge_reverse(self, vertex_to: Vertex) -> list[Vertex]:
        """Returns vertices directing to a particular vertex

        Args:
            vertex_to (Vertex): the vertex to go to

        Returns:
            list[Vertex]: the vertices leading to that vertex
        """
        return [
            vertex_from
            for vertex_from in self._edges
            if vertex_to in self._edges[vertex_from]
        ]

    def weight(self, vertex_to: Vertex, vertex_from: Vertex) -> float:
        """Returns the weight of an edge

        Args:
            vertex_to (Vertex): the first vertex of the edge
            vertex_from (Vertex): the second vertex of the edge

        Returns:
            float: the weight of the edge
        """
        return self._edges[vertex_from][vertex_to]

    def order(self) -> int:
        """Gets the number of vertices

        Returns:
            int: the number of vertices
        """
        return len(self._edges)

    def visit(self, vertex: Vertex):
        """Sets a vertex as visited

        Args:
            vertex (Vertex): the vertex to visit
        """
        self._vertices[vertex] = True

    def is_visited(self, vertex: Vertex) -> bool:
        """Gets the status of a given vertex

        Args:
            vertex (Vertex): the vertex

        Returns:
            bool: True is visited, else False
        """
        return self._vertices[vertex]

    def route(self) -> dict[Vertex, bool]:
        """Gets the vertices and if they were visited

        Returns:
            dict[Vertex, bool]: dictionnary of the vertices containing the vertices as keys and
            a boolean indicating if the vertex has been visited
        """
        return self._vertices

    def degree(self, vertex: Vertex) -> int:
        """Gets the degree of a given vertex

        Args:
            vertex (Vertex): the vertex

        Returns:
            int: the degree of the vertex
        """
        return len(self._edges[vertex])

    def remove(self, vertex: Vertex):
        """Removes a particular vertex and its underlying edges

        Args:
            vertex (Vertex): the vertex to remove
        """
        for vertex_to in self._edges[vertex].keys():
            self._edges[vertex_to].pop(vertex)
        self._edges.pop(vertex)
        self._vertices.pop(vertex)

    def disconnect(self, vertex_from: Vertex, vertex_to: Vertex):
        """Removes a particular edge

        Args:
            vertex_from (Vertex): the first vertex of the edge
            vertex_to (Vertex): the second vertex of the edge
        """
        del self._edges[vertex_from][vertex_to]

    def clear(self, vertex: Optional[Vertex] = None):
        """Clear a visited vertex, or all vertices if None is provided

        Args:
            vertex (Optional[Vertex], optional): the vertex to clear. Defaults to None.
        """
        if vertex is None:
            self._vertices = dict(
                zip(self._edges.keys(), [False for i in range(len(self._edges))])
            )
        elif vertex:
            self._vertices[vertex] = False
