import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cma_es import CMAES
from functions import get_function


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    sampler_type: str
    function_name: str
    dimension: int
    n_runs: int
    max_evals: int
    target_fitness: float
    random_seeds: list[int]


@dataclass
class RunResult:
    """Results from a single optimization run."""
    sampler_type: str
    function_name: str
    dimension: int
    run_id: int
    seed: int
    best_fitness: float
    evaluations: int
    generations: int
    execution_time: float
    converged: bool
    fitness_history: list[float]
    distance_to_optimum: float


class ExperimentFramework:
    """Framework for running systematic CMA-ES experiments."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.ensure_results_dir()

    def ensure_results_dir(self):
        """Create results directory if it doesn't exist."""
        os.makedirs(self.results_dir, exist_ok=True)

    def create_standard_config(self) -> list[ExperimentConfig]:
        """Create standard experimental configuration for the project."""
        configs = []
        functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'schwefel',
                     'griewank', 'levy', 'zakharov', 'michalewicz', 'beale', 'booth', 'matyas']
        samplers = ['gaussian', 'sobol', 'halton']
        dimensions = [10, 20]

        for sampler in samplers:
            for func_name in functions:
                for dim in dimensions:
                    if func_name in ['beale', 'booth', 'matyas'] and dim != 2:
                        continue

                    config = ExperimentConfig(
                        sampler_type=sampler,
                        function_name=func_name,
                        dimension=dim,
                        n_runs=30,
                        max_evals=10000 * dim,
                        target_fitness=1e-8,
                        random_seeds=list(range(42, 42 + 30))
                    )
                    configs.append(config)

        return configs

    def run_single_experiment(self, config: ExperimentConfig) -> list[RunResult]:
        """Run a single experiment configuration with multiple runs."""

        print(f"Ejecutando: {config.sampler_type} en {config.function_name} ({config.dimension}D)")

        func = get_function(config.function_name, config.dimension)
        results = []

        for run_id, seed in enumerate(config.random_seeds):
            print(f"  Ejecución {run_id + 1}/{config.n_runs} (semilla: {seed})", end=" ")

            np.random.seed(seed)

            cma = CMAES(dimension=config.dimension, sampler_type=config.sampler_type)

            start_time = time.time()

            result = cma.optimize(
                objective_func=func,
                max_evals=config.max_evals,
                target_fitness=config.target_fitness,
                verbose=False
            )

            execution_time = time.time() - start_time

            distance = np.linalg.norm(result['best_solution'] - func.global_optimum_location)

            run_result = RunResult(
                sampler_type=config.sampler_type,
                function_name=config.function_name,
                dimension=config.dimension,
                run_id=run_id,
                seed=seed,
                best_fitness=result['best_fitness'],
                evaluations=result['evaluations'],
                generations=result['generations'],
                execution_time=execution_time,
                converged=result['converged'],
                fitness_history=result['fitness_history'],
                distance_to_optimum=distance
            )

            results.append(run_result)
            print(f"OK {result['best_fitness']:.2e} ({result['evaluations']} evals)")

        return results

    def run_all_experiments(self, configs: list[ExperimentConfig] = None) -> list[RunResult]:
        """Run all experiments and return comprehensive results."""

        if configs is None:
            configs = self.create_standard_config()

        print(f"Iniciando suite experimental: {len(configs)} configuraciones")
        print(f"Total de ejecuciones estimadas: {sum(c.n_runs for c in configs)}")
        print("-" * 60)

        all_results = []

        for i, config in enumerate(configs):
            print(f"Configuración {i + 1}/{len(configs)}:")

            results = self.run_single_experiment(config)
            all_results.extend(results)

            self.save_results(all_results, f"intermediate_results.json")

        print(f"Suite experimental completado!")
        print(f"Total de ejecuciones exitosas: {len(all_results)}")

        return all_results

    def save_results(self, results: list[RunResult], filename: str):
        """Save results to JSON file."""

        serializable_results = []
        for result in results:
            result_dict = {
                'sampler_type': result.sampler_type,
                'function_name': result.function_name,
                'dimension': result.dimension,
                'run_id': result.run_id,
                'seed': result.seed,
                'best_fitness': float(result.best_fitness),
                'evaluations': int(result.evaluations),
                'generations': int(result.generations),
                'execution_time': float(result.execution_time),
                'converged': bool(result.converged),
                'fitness_history': [float(x) for x in result.fitness_history],
                'distance_to_optimum': float(result.distance_to_optimum)
            }
            serializable_results.append(result_dict)

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Resultados guardados en: {filepath}")

    def load_results(self, filename: str) -> list[RunResult]:
        """Load results from JSON file."""

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = []
        for item in data:
            result = RunResult(
                sampler_type=item['sampler_type'],
                function_name=item['function_name'],
                dimension=item['dimension'],
                run_id=item['run_id'],
                seed=item['seed'],
                best_fitness=item['best_fitness'],
                evaluations=item['evaluations'],
                generations=item['generations'],
                execution_time=item['execution_time'],
                converged=item['converged'],
                fitness_history=item['fitness_history'],
                distance_to_optimum=item['distance_to_optimum']
            )
            results.append(result)

        return results

    def results_to_dataframe(self, results: list[RunResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""

        data = []
        for result in results:
            row = {
                'sampler_type': result.sampler_type,
                'function_name': result.function_name,
                'dimension': result.dimension,
                'run_id': result.run_id,
                'seed': result.seed,
                'best_fitness': result.best_fitness,
                'evaluations': result.evaluations,
                'generations': result.generations,
                'execution_time': result.execution_time,
                'converged': result.converged,
                'distance_to_optimum': result.distance_to_optimum
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_summary_statistics(self, results: list[RunResult]) -> pd.DataFrame:
        """Generate summary statistics for each sampler-function combination."""

        df = self.results_to_dataframe(results)

        grouped = df.groupby(['sampler_type', 'function_name', 'dimension'])

        summary = grouped.agg({
            'best_fitness': ['mean', 'std', 'min', 'max', 'median'],
            'evaluations': ['mean', 'std', 'min', 'max', 'median'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'converged': ['sum', 'count'],
            'distance_to_optimum': ['mean', 'std', 'min', 'max']
        }).round(6)

        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary['success_rate'] = summary['converged_sum'] / summary['converged_count']

        return summary


def run_quick_test():
    """Run a quick test with reduced parameters."""

    framework = ExperimentFramework()

    quick_configs = []
    for sampler in ['gaussian', 'sobol', 'halton']:
        for func_name in ['sphere', 'rastrigin']:
            config = ExperimentConfig(
                sampler_type=sampler,
                function_name=func_name,
                dimension=10,
                n_runs=3,
                max_evals=1000,
                target_fitness=1e-8,
                random_seeds=[42, 43, 44]
            )
            quick_configs.append(config)

    print("Ejecutando prueba rápida...")
    results = framework.run_all_experiments(quick_configs)

    framework.save_results(results, "quick_test_results.json")

    summary = framework.get_summary_statistics(results)
    print("Resumen de prueba rápida:")
    print(summary[['best_fitness_mean', 'evaluations_mean', 'success_rate']])

    return results
