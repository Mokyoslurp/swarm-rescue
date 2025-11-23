from dataclasses import dataclass


@dataclass
class Vertex:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Vertex):
            return self.x == __value.x and self.y == __value.y
        return False

    def __add__(self, other: object):
        if isinstance(other, Vertex):
            return Vertex(x=self.x + other.x, y=self.y + other.y)

        raise TypeError

    def __radd__(self, other: object):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other: object):
        if isinstance(other, Vertex):
            return Vertex(x=round(self.x / other.x), y=round(self.x / other.y))
        elif isinstance(other, float) or isinstance(other, int):
            return Vertex(x=round(self.x / other), y=round(self.y / other))

        raise TypeError

    def __str__(self) -> str:
        return f"X = {self.x} ; Y = {self.y}"
