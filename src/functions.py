from typing import Tuple

import numpy as np


class BenchmarkFunction:
    """Base class for benchmark optimization functions."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.name = self.__class__.__name__

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate function at point x."""
        if len(x) != self.dimension:
            raise ValueError(f"Expected {self.dimension}D input, got {len(x)}D")
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> float:
        """Override this method in subclasses."""
        raise NotImplementedError

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return (lower_bound, upper_bound) for variables."""
        raise NotImplementedError

    @property
    def global_optimum(self) -> float:
        """Return global optimum value."""
        raise NotImplementedError

    @property
    def global_optimum_location(self) -> np.ndarray:
        """Return location of global optimum."""
        raise NotImplementedError


class Sphere(BenchmarkFunction):
    """Sphere function: f(x) = sum(x_i^2)

    Global minimum: f(0, ..., 0) = 0
    Domain: [-5.12, 5.12]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.12, 5.12)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.zeros(self.dimension)


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function: f(x) = sum(100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Global minimum: f(1, ..., 1) = 0
    Domain: [-2.048, 2.048]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-2.048, 2.048)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.ones(self.dimension)


class Rastrigin(BenchmarkFunction):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))

    Global minimum: f(0, ..., 0) = 0
    Domain: [-5.12, 5.12]^n
    """

    def __init__(self, dimension: int, A: float = 10.0):
        super().__init__(dimension)
        self.A = A

    def evaluate(self, x: np.ndarray) -> float:
        return self.A * self.dimension + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.12, 5.12)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.zeros(self.dimension)

# Factory function for easy creation


def get_function(name: str, dimension: int) -> BenchmarkFunction:
    """Get benchmark function by name."""
    functions = {
        'sphere': Sphere,
        'rosenbrock': Rosenbrock,
        'rastrigin': Rastrigin
    }

    if name.lower() not in functions:
        raise ValueError(f"Unknown function: {name}. Available: {list(functions.keys())}")

    return functions[name.lower()](dimension)
