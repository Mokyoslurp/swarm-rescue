import math
from swarm_rescue.solutions.my_solution.planning.graph import GraphADT, Vertex


def a_star(adt: GraphADT, start: Vertex, end: Vertex) -> tuple[float, list[Vertex]]:
    """A* Algorithm,
    a generalized Dijkstra's algorithm with heuristic function to reduce execution time"""

    # All weights in dcg must be positive
    # Otherwise we have to use Bellman Ford instead
    negative_check = [
        weight
        for vertex_from in adt.edges
        for weight in adt.edges[vertex_from].values()
    ]
    assert min(negative_check) >= 0, (
        "Negative weights are not allowed, please use Bellman-Ford"
    )

    # queue is used to check the vertex with the minimum summation
    queue: dict[Vertex, float] = {}
    queue[start] = 0

    # distance keeps track of distance from starting vertex to any vertex
    distance: dict[Vertex, float] = {}

    # heuristic keeps track of distance from ending vertex to any vertex
    heuristic: dict[Vertex, float] = {}

    # route is a dict of the summation of distance and heuristic
    route: dict[Vertex, float] = {}

    # criteria
    for vertex in adt.vertices:
        # initialize
        distance[vertex] = float("inf")

        # Absolute distance
        heuristic[vertex] = math.sqrt((vertex.x - end.x) ** 2 + (vertex.y - end.y) ** 2)

    # initialize
    distance[start] = 0

    # pred keeps track of how we get to the current vertex
    pred = {}

    # dynamic programming
    path = []
    while queue:
        # vertex with the minimum summation
        current_vertex = min(queue, key=queue.get)  # type: ignore
        queue.pop(current_vertex)

        # find the minimum summation of both distances
        minimum = float("inf")

        for vertex_to in adt.edge(current_vertex):
            # check if the current vertex can construct the optimal path
            # from the beginning and to the end
            distance[vertex_to] = distance[current_vertex] + adt.weight(
                current_vertex, vertex_to
            )
            route[vertex_to] = distance[vertex_to] + heuristic[vertex_to]
            if route[vertex_to] < minimum:
                minimum = route[vertex_to]

            # only append unvisited and unqueued vertices
            if (not adt.is_visited(vertex_to)) and (vertex_to not in queue):
                queue[vertex_to] = route[vertex_to]
                pred[vertex_to] = current_vertex

        # each vertex is visited only once
        adt.visit(current_vertex)

        # traversal ends when the target is met
        if current_vertex == end:
            # create the shortest path by backtracking
            # trace the predecessor vertex from end to start
            # print("Current == End")
            previous = end
            while pred:
                path.insert(0, previous)
                if previous == start:
                    break
                previous = pred[previous]
            break

    # print("Path :", path)
    if len(path) == 0:
        return None, path
    # note that if we cant go from start to end
    # we may get inf for distance[end]
    # additionally, the path may not include start position
    return distance[end], path
