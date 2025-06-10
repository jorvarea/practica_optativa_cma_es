import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from experiments import ExperimentFramework, RunResult


@dataclass
class StatisticalTest:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    interpretation: str


@dataclass
class ComparisonResult:
    """Results from comparing two samplers."""
    sampler1: str
    sampler2: str
    function_name: str
    dimension: int
    metric: str
    test_result: StatisticalTest
    effect_size: float
    winner: str


class StatisticalAnalyzer:
    """Statistical analysis for CMA-ES sampling comparison."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test_normality(self, data: np.ndarray, test_name: str = "Shapiro-Wilk") -> StatisticalTest:
        """Test normality of data distribution."""

        if len(data) < 3:
            return StatisticalTest(
                test_name=test_name,
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Muestra muy pequeña para test de normalidad"
            )

        if test_name.lower() == "shapiro-wilk":
            # Shapiro-Wilk test (good for small samples)
            if len(data) > 5000:
                # For large samples, use Kolmogorov-Smirnov instead
                statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
                test_name = "Kolmogorov-Smirnov"
            else:
                statistic, p_value = stats.shapiro(data)
        else:
            raise ValueError(f"Unknown normality test: {test_name}")

        significant = p_value < self.alpha

        if significant:
            interpretation = f"Los datos NO siguen distribución normal (p={p_value:.4f})"
        else:
            interpretation = f"Los datos siguen distribución normal (p={p_value:.4f})"

        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interpretation
        )

    def wilcoxon_signed_rank_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTest:
        """Wilcoxon signed-rank test for paired samples."""

        if len(data1) != len(data2):
            raise ValueError("Data arrays must have the same length for paired test")

        if len(data1) < 6:
            return StatisticalTest(
                test_name="Wilcoxon Signed-Rank",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for Wilcoxon test (need ≥6 pairs)"
            )

        # Remove tied pairs (differences of zero)
        differences = data1 - data2
        non_zero_diffs = differences[differences != 0]

        if len(non_zero_diffs) < 6:
            return StatisticalTest(
                test_name="Wilcoxon Signed-Rank",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Too many tied pairs for Wilcoxon test"
            )

        statistic, p_value = stats.wilcoxon(non_zero_diffs)
        significant = p_value < self.alpha

        if significant:
            interpretation = f"Significant difference between paired samples (p={p_value:.4f} < α={self.alpha})"
        else:
            interpretation = f"No significant difference between paired samples (p={p_value:.4f} ≥ α={self.alpha})"

        return StatisticalTest(
            test_name="Wilcoxon Signed-Rank",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interpretation
        )

    def kruskal_wallis_test(self, *groups: np.ndarray) -> StatisticalTest:
        """Kruskal-Wallis test for multiple independent groups."""

        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for Kruskal-Wallis test")

        # Filter out empty groups
        valid_groups = [group for group in groups if len(group) > 0]

        if len(valid_groups) < 2:
            return StatisticalTest(
                test_name="Kruskal-Wallis",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient valid groups for Kruskal-Wallis test"
            )

        # Check minimum sample sizes
        min_size = min(len(group) for group in valid_groups)
        if min_size < 5:
            return StatisticalTest(
                test_name="Kruskal-Wallis",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data in one or more groups (need ≥5 per group)"
            )

        statistic, p_value = stats.kruskal(*valid_groups)
        significant = p_value < self.alpha

        if significant:
            interpretation = f"Significant difference between groups (p={p_value:.4f} < α={self.alpha})"
        else:
            interpretation = f"No significant difference between groups (p={p_value:.4f} ≥ α={self.alpha})"

        return StatisticalTest(
            test_name="Kruskal-Wallis",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interpretation
        )

    def mann_whitney_u_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTest:
        """Mann-Whitney U test for independent samples."""

        if len(data1) < 3 or len(data2) < 3:
            return StatisticalTest(
                test_name="Mann-Whitney U",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for Mann-Whitney U test (need ≥3 per group)"
            )

        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        significant = p_value < self.alpha

        if significant:
            interpretation = f"Significant difference between independent samples (p={p_value:.4f} < α={self.alpha})"
        else:
            interpretation = f"No significant difference between independent samples (p={p_value:.4f} ≥ α={self.alpha})"

        return StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interpretation
        )

    def calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray, method: str = "cohen_d") -> float:
        """Calculate effect size between two groups."""

        if method.lower() == "cohen_d":
            # Cohen's d for effect size
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

            # Pooled standard deviation
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

            if pooled_std == 0:
                return 0.0

            effect_size = (mean1 - mean2) / pooled_std
            return abs(effect_size)

        elif method.lower() == "rank_biserial":
            # Rank-biserial correlation for Mann-Whitney U
            try:
                u_stat, _ = stats.mannwhitneyu(data1, data2)
                n1, n2 = len(data1), len(data2)
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                return abs(effect_size)
            except:
                return 0.0

        else:
            raise ValueError(f"Unknown effect size method: {method}")

    def compare_samplers_on_function(self, results: list[RunResult], function_name: str, dimension: int, metric: str = "best_fitness") -> list[ComparisonResult]:
        """Compare all sampler pairs on a specific function."""

        # Filter results for this function and dimension
        filtered_results = [r for r in results
                            if r.function_name == function_name and r.dimension == dimension]

        if not filtered_results:
            return []

        # Group by sampler
        sampler_data = {}
        for result in filtered_results:
            if result.sampler_type not in sampler_data:
                sampler_data[result.sampler_type] = []

            # Extract the specified metric
            if metric == "best_fitness":
                value = result.best_fitness
            elif metric == "evaluations":
                value = result.evaluations
            elif metric == "execution_time":
                value = result.execution_time
            elif metric == "distance_to_optimum":
                value = result.distance_to_optimum
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sampler_data[result.sampler_type].append(value)

        # Convert to numpy arrays
        for sampler in sampler_data:
            sampler_data[sampler] = np.array(sampler_data[sampler])

        samplers = list(sampler_data.keys())
        comparisons = []

        # Pairwise comparisons
        for i in range(len(samplers)):
            for j in range(i + 1, len(samplers)):
                sampler1, sampler2 = samplers[i], samplers[j]
                data1, data2 = sampler_data[sampler1], sampler_data[sampler2]

                # Choose appropriate test
                if len(data1) == len(data2):
                    # Paired test (same seeds were used)
                    test_result = self.wilcoxon_signed_rank_test(data1, data2)
                    effect_size = self.calculate_effect_size(data1, data2, "cohen_d")
                else:
                    # Independent test
                    test_result = self.mann_whitney_u_test(data1, data2)
                    effect_size = self.calculate_effect_size(data1, data2, "rank_biserial")

                # Determine winner (for minimization problems)
                if metric in ["best_fitness", "evaluations", "execution_time", "distance_to_optimum"]:
                    # Lower is better
                    winner = sampler1 if np.mean(data1) < np.mean(data2) else sampler2
                else:
                    # Higher is better
                    winner = sampler1 if np.mean(data1) > np.mean(data2) else sampler2

                comparison = ComparisonResult(
                    sampler1=sampler1,
                    sampler2=sampler2,
                    function_name=function_name,
                    dimension=dimension,
                    metric=metric,
                    test_result=test_result,
                    effect_size=effect_size,
                    winner=winner
                )

                comparisons.append(comparison)

        return comparisons

    def comprehensive_analysis(self, results: list[RunResult]) -> dict:
        """Run comprehensive statistical analysis on all results."""

        print("Ejecutando análisis estadístico comprehensivo...")
        print("-" * 50)

        framework = ExperimentFramework()
        df = framework.results_to_dataframe(results)

        analysis_results = {
            'normality_tests': [],
            'pairwise_comparisons': [],
            'multiple_comparisons': []
        }

        functions = df['function_name'].unique()
        dimensions = df['dimension'].unique()
        samplers = df['sampler_type'].unique()

        for func in functions:
            for dim in dimensions:
                func_dim_data = df[(df['function_name'] == func) & (df['dimension'] == dim)]

                if func_dim_data.empty:
                    continue

                for sampler in samplers:
                    sampler_data = func_dim_data[func_dim_data['sampler_type'] == sampler]['best_fitness'].values

                    if len(sampler_data) > 0:
                        normality_test = self.test_normality(sampler_data)
                        analysis_results['normality_tests'].append({
                            'function': func,
                            'dimension': dim,
                            'sampler': sampler,
                            'test': normality_test
                        })

                for metric in ['best_fitness']:
                    comparisons = self.compare_samplers_on_function(results, func, dim, metric)
                    analysis_results['pairwise_comparisons'].extend(comparisons)

                sampler_groups = []
                for sampler in samplers:
                    sampler_data = func_dim_data[func_dim_data['sampler_type'] == sampler]['best_fitness'].values
                    if len(sampler_data) > 0:
                        sampler_groups.append(sampler_data)

                if len(sampler_groups) >= 2:
                    kruskal_test = self.kruskal_wallis_test(*sampler_groups)
                    analysis_results['multiple_comparisons'].append({
                        'function': func,
                        'dimension': dim,
                        'test': kruskal_test
                    })

                    print(f"  {func} ({dim}D): {kruskal_test.interpretation}")

        return analysis_results

    def generate_statistical_report(self, analysis_results: dict) -> str:
        """Generate a formatted statistical report."""

        report = []
        report.append("-" * 60)
        report.append("REPORTE DE ANÁLISIS ESTADÍSTICO")
        report.append("-" * 60)

        report.append(f"TESTS DE NORMALIDAD ({len(analysis_results['normality_tests'])} tests)")
        for test_data in analysis_results['normality_tests']:
            test = test_data['test']
            report.append(f"  {test_data['function']} ({test_data['dimension']}D) - {test_data['sampler']}:")
            report.append(f"    {test.interpretation}")

        report.append(f"COMPARACIONES PAREADAS ({len(analysis_results['pairwise_comparisons'])} comparaciones)")
        significant_comparisons = [
            comp for comp in analysis_results['pairwise_comparisons'] if comp.test_result.significant]

        if significant_comparisons:
            report.append(f"  Encontradas {len(significant_comparisons)} diferencias significativas:")
            for comp in significant_comparisons:
                report.append(f"    {comp.function_name} ({comp.dimension}D): {comp.sampler1} vs {comp.sampler2}")
                report.append(f"      Ganador: {comp.winner} (p={comp.test_result.p_value:.4f})")
        else:
            report.append("  No se encontraron diferencias significativas entre muestreadores.")

        report.append(
            f"COMPARACIONES DE MÚLTIPLES GRUPOS ({len(analysis_results.get('multiple_comparisons', []))} tests)")
        for test_data in analysis_results.get('multiple_comparisons', []):
            test = test_data['test']
            report.append(f"  {test_data['function']} ({test_data['dimension']}D): {test.interpretation}")

        report.append("-" * 60)

        return "\n".join(report)


def run_statistical_analysis_test():
    """Test the statistical analysis with quick test data."""

    print("Probando Módulo de Análisis Estadístico")
    print("-" * 50)

    framework = ExperimentFramework()
    results = framework.load_results("quick_test_results.json")
    print(f"Cargados {len(results)} resultados de prueba rápida")

    analyzer = StatisticalAnalyzer()
    analysis_results = analyzer.comprehensive_analysis(results)

    print(f"Análisis completado:")
    print(f"  - Tests de normalidad: {len(analysis_results['normality_tests'])}")
    print(f"  - Comparaciones pareadas: {len(analysis_results['pairwise_comparisons'])}")
    print(f"  - Comparaciones múltiples: {len(analysis_results.get('multiple_comparisons', []))}")

    report = analyzer.generate_statistical_report(analysis_results)
    print("Reporte estadístico generado:")
    print(report[:500] + "..." if len(report) > 500 else report)

    print("Prueba de análisis estadístico completada exitosamente!")

    return analysis_results
