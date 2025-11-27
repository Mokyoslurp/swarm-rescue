from swarm_rescue.solutions.my_solution.mapping.vertex import Vertex


class Frontier:
    def __init__(self, points: list[Vertex]):
        self.points = points

    def barycenter(self) -> Vertex:
        return sum(self.points) / len(self.points)

    def length_squared(self) -> float:
        coordinates = [(vertex.x, vertex.y) for vertex in self.points]
        x, y = zip(*coordinates)
        return (max(x) - min(x)) ** 2 + (max(y) - min(y)) ** 2

    def add_point(self, point: Vertex):
        self.points.append(point)

    def is_close_to_point(self, vertex: Vertex) -> bool:
        if vertex in self.points:
            return True

        x = vertex.x
        y = vertex.y

        combinations = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
        ]
        for i, j in combinations:
            if Vertex(i, j) in self.points:
                return True

        return False

    def __str__(self) -> str:
        return str(self.barycenter())
