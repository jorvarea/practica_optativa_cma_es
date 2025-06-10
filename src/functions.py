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


class Ackley(BenchmarkFunction):
    """Ackley function: f(x) = -a*exp(-b*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + e

    Global minimum: f(0, ..., 0) = 0
    Domain: [-32.768, 32.768]^n
    """

    def __init__(self, dimension: int, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        super().__init__(dimension)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: np.ndarray) -> float:
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(self.c * x)))
        return term1 + term2 + self.a + np.e

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-32.768, 32.768)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.zeros(self.dimension)


class Schwefel(BenchmarkFunction):
    """Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))

    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    Domain: [-500, 500]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        return 418.9829 * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-500.0, 500.0)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.full(self.dimension, 420.9687)


class Griewank(BenchmarkFunction):
    """Griewank function: f(x) = 1/4000 * sum(x_i^2) - prod(cos(x_i/sqrt(i+1))) + 1

    Global minimum: f(0, ..., 0) = 0
    Domain: [-600, 600]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1))))
        return sum_term - prod_term + 1

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-600.0, 600.0)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.zeros(self.dimension)


class Levy(BenchmarkFunction):
    """Levy function: Complex multimodal function

    Global minimum: f(1, ..., 1) = 0
    Domain: [-10, 10]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        w = 1 + (x - 1) / 4

        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        return term1 + term2 + term3

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.ones(self.dimension)


class Zakharov(BenchmarkFunction):
    """Zakharov function: f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4

    Global minimum: f(0, ..., 0) = 0
    Domain: [-5, 10]^n
    """

    def evaluate(self, x: np.ndarray) -> float:
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, self.dimension + 1) * x)
        return sum1 + sum2**2 + sum2**4

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.0, 10.0)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.zeros(self.dimension)


class Michalewicz(BenchmarkFunction):
    """Michalewicz function: f(x) = -sum(sin(x_i) * sin(i*x_i^2/pi)^(2*m))

    Global minimum: Variable (depends on dimension)
    Domain: [0, pi]^n
    """

    def __init__(self, dimension: int, m: float = 10):
        super().__init__(dimension)
        self.m = m

    def evaluate(self, x: np.ndarray) -> float:
        i = np.arange(1, self.dimension + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * self.m))

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0.0, np.pi)

    @property
    def global_optimum(self) -> float:
        # Approximate values for common dimensions
        if self.dimension == 2:
            return -1.8013
        elif self.dimension == 5:
            return -4.687658
        elif self.dimension == 10:
            return -9.66015
        else:
            return -self.dimension * 0.966  # Rough approximation

    @property
    def global_optimum_location(self) -> np.ndarray:
        # Approximation - exact values are complex
        return np.full(self.dimension, np.pi / 2)


class Beale(BenchmarkFunction):
    """Beale function: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Global minimum: f(3, 0.5) = 0
    Domain: [-4.5, 4.5]^2
    Note: Only defined for 2D
    """

    def __init__(self, dimension: int = 2):
        if dimension != 2:
            raise ValueError("Beale function is only defined for 2 dimensions")
        super().__init__(dimension)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1 * x2)**2
        term2 = (2.25 - x1 + x1 * x2**2)**2
        term3 = (2.625 - x1 + x1 * x2**3)**2
        return term1 + term2 + term3

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-4.5, 4.5)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.array([3.0, 0.5])


class Booth(BenchmarkFunction):
    """Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2

    Global minimum: f(1, 3) = 0
    Domain: [-10, 10]^2
    Note: Only defined for 2D
    """

    def __init__(self, dimension: int = 2):
        if dimension != 2:
            raise ValueError("Booth function is only defined for 2 dimensions")
        super().__init__(dimension)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)

    @property
    def global_optimum(self) -> float:
        return 0.0

    @property
    def global_optimum_location(self) -> np.ndarray:
        return np.array([1.0, 3.0])


class Matyas(BenchmarkFunction):
    """Matyas function: f(x,y) = 0.26(x^2 + y^2) - 0.48xy

    Global minimum: f(0, 0) = 0
    Domain: [-10, 10]^2
    Note: Only defined for 2D
    """

    def __init__(self, dimension: int = 2):
        if dimension != 2:
            raise ValueError("Matyas function is only defined for 2 dimensions")
        super().__init__(dimension)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)

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
        'rastrigin': Rastrigin,
        'ackley': Ackley,
        'schwefel': Schwefel,
        'griewank': Griewank,
        'levy': Levy,
        'zakharov': Zakharov,
        'michalewicz': Michalewicz,
        'beale': Beale,
        'booth': Booth,
        'matyas': Matyas
    }

    if name.lower() not in functions:
        raise ValueError(f"Unknown function: {name}. Available: {list(functions.keys())}")

    # Handle 2D-only functions
    if name.lower() in ['beale', 'booth', 'matyas']:
        return functions[name.lower()]()
    else:
        return functions[name.lower()](dimension)
