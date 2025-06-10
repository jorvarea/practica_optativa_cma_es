import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments import ExperimentFramework, RunResult
from statistical_analysis import StatisticalAnalyzer


class VisualizationGenerator:
    """Generate comprehensive visualizations for CMA-ES sampling comparison."""

    def __init__(self, output_dir: str = "plots", style: str = "seaborn-v0_8"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set matplotlib style
        plt.style.use('default')  # Use default since seaborn-v0_8 might not be available
        sns.set_palette("husl")

        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.dpi': 100
        })

    def plot_convergence_curves(self, results: List[RunResult],
                                function_name: str, dimension: int,
                                save_plot: bool = True) -> None:
        """Plot convergence curves for all samplers on a specific function."""

        # Filter results for this function and dimension
        filtered_results = [r for r in results
                            if r.function_name == function_name and r.dimension == dimension]

        if not filtered_results:
            print(f"No results found for {function_name} ({dimension}D)")
            return

        plt.figure(figsize=(12, 8))

        # Group by sampler
        sampler_colors = {'gaussian': 'blue', 'sobol': 'red', 'halton': 'green'}

        for sampler_type in ['gaussian', 'sobol', 'halton']:
            sampler_results = [r for r in filtered_results if r.sampler_type == sampler_type]

            if not sampler_results:
                continue

            # Plot individual runs (light colors)
            for i, result in enumerate(sampler_results[:5]):  # Show first 5 runs to avoid clutter
                if result.fitness_history:
                    generations = range(len(result.fitness_history))
                    color = sampler_colors.get(sampler_type, 'black')
                    alpha = 0.3
                    label = sampler_type.title() if i == 0 else None
                    plt.semilogy(generations, result.fitness_history,
                                 color=color, alpha=alpha, linewidth=1, label=label)

            # Plot average convergence (bold lines)
            if sampler_results:
                # Find the minimum length to align all histories
                min_length = min(len(r.fitness_history) for r in sampler_results if r.fitness_history)

                if min_length > 0:
                    # Truncate all histories to the same length and compute mean
                    aligned_histories = []
                    for result in sampler_results:
                        if result.fitness_history and len(result.fitness_history) >= min_length:
                            aligned_histories.append(result.fitness_history[:min_length])

                    if aligned_histories:
                        mean_history = np.mean(aligned_histories, axis=0)
                        generations = range(len(mean_history))
                        color = sampler_colors.get(sampler_type, 'black')
                        plt.semilogy(generations, mean_history,
                                     color=color, linewidth=3, alpha=0.8,
                                     label=f'{sampler_type.title()} (Average)')

        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'Convergence Curves: {function_name.title()} Function ({dimension}D)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            filename = f"convergence_{function_name}_{dimension}D.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot: {filepath}")

        plt.close()  # Close figure to free memory

    def plot_performance_boxplots(self, results: List[RunResult],
                                  metric: str = "best_fitness",
                                  save_plot: bool = True) -> None:
        """Plot boxplots comparing sampler performance across all functions."""

        framework = ExperimentFramework()
        df = framework.results_to_dataframe(results)

        if df.empty:
            print("No results to plot")
            return

        # Get unique functions and dimensions
        functions = df['function_name'].unique()
        dimensions = sorted(df['dimension'].unique())

        # Create subplots for each dimension
        fig, axes = plt.subplots(len(dimensions), 1, figsize=(15, 6 * len(dimensions)))
        if len(dimensions) == 1:
            axes = [axes]

        for dim_idx, dim in enumerate(dimensions):
            ax = axes[dim_idx]

            # Filter data for this dimension
            dim_data = df[df['dimension'] == dim]

            # Prepare data for plotting
            plot_data = []
            positions = []
            labels = []

            pos = 0
            for func in sorted(dim_data['function_name'].unique()):
                func_data = dim_data[dim_data['function_name'] == func]

                for sampler in ['gaussian', 'sobol', 'halton']:
                    sampler_data = func_data[func_data['sampler_type'] == sampler][metric].values
                    if len(sampler_data) > 0:
                        plot_data.append(sampler_data)
                        positions.append(pos)
                        labels.append(f"{func}\n{sampler}")
                        pos += 1
                pos += 0.5  # Space between functions

            # Create boxplot
            if plot_data:
                bp = ax.boxplot(plot_data, positions=positions, patch_artist=True)

                # Color boxes by sampler
                colors = {'gaussian': 'lightblue', 'sobol': 'lightcoral', 'halton': 'lightgreen'}
                for i, (patch, label) in enumerate(zip(bp['boxes'], labels)):
                    sampler = label.split('\n')[1]
                    patch.set_facecolor(colors.get(sampler, 'lightgray'))

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Performance Comparison ({dim}D) - {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

            # Set log scale for fitness metrics
            if metric in ['best_fitness', 'distance_to_optimum']:
                ax.set_yscale('log')

        plt.tight_layout()

        if save_plot:
            filename = f"boxplots_{metric}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot: {filepath}")

        plt.close()

    def plot_performance_heatmap(self, results: List[RunResult],
                                 metric: str = "best_fitness",
                                 save_plot: bool = True) -> None:
        """Plot heatmap of average performance across functions and samplers."""

        framework = ExperimentFramework()
        df = framework.results_to_dataframe(results)

        if df.empty:
            print("No results to plot")
            return

        # Create separate heatmaps for each dimension
        dimensions = sorted(df['dimension'].unique())

        for dim in dimensions:
            dim_data = df[df['dimension'] == dim]

            # Pivot table for heatmap
            pivot_table = dim_data.pivot_table(
                values=metric,
                index='function_name',
                columns='sampler_type',
                aggfunc='mean'
            )

            # Reorder columns
            desired_order = ['gaussian', 'sobol', 'halton']
            available_columns = [col for col in desired_order if col in pivot_table.columns]
            pivot_table = pivot_table[available_columns]

            plt.figure(figsize=(8, 10))

            # Use log scale for fitness metrics
            if metric in ['best_fitness', 'distance_to_optimum']:
                # Take log for better visualization
                log_data = np.log10(pivot_table + 1e-15)  # Add small value to avoid log(0)
                sns.heatmap(log_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                            cbar_kws={'label': f'log10({metric.replace("_", " ").title()})'})
            else:
                sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r',
                            cbar_kws={'label': metric.replace('_', ' ').title()})

            plt.title(f'Average {metric.replace("_", " ").title()} Heatmap ({dim}D)')
            plt.xlabel('Sampler Type')
            plt.ylabel('Function')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            if save_plot:
                filename = f"heatmap_{metric}_{dim}D.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved heatmap: {filepath}")

            plt.close()

    def plot_statistical_significance(self, analysis_results: Dict[str, Any],
                                      save_plot: bool = True) -> None:
        """Plot statistical significance results."""

        if 'pairwise_comparisons' not in analysis_results:
            print("No pairwise comparisons found in analysis results")
            return

        significant_comparisons = [comp for comp in analysis_results['pairwise_comparisons']
                                   if comp.test_result.significant]

        if not significant_comparisons:
            print("No significant differences found to plot")
            return

        # Prepare data for visualization
        functions = list(set(comp.function_name for comp in significant_comparisons))
        dimensions = list(set(comp.dimension for comp in significant_comparisons))

        fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 6 * len(dimensions)))
        if len(dimensions) == 1:
            axes = [axes]

        for dim_idx, dim in enumerate(dimensions):
            ax = axes[dim_idx]

            # Filter comparisons for this dimension
            dim_comparisons = [comp for comp in significant_comparisons
                               if comp.dimension == dim]

            if not dim_comparisons:
                ax.text(0.5, 0.5, f'No significant differences\nfor {dim}D functions',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Statistical Significance ({dim}D)')
                continue

            # Create matrix of p-values
            func_names = sorted(set(comp.function_name for comp in dim_comparisons))
            sampler_pairs = sorted(set(f"{comp.sampler1} vs {comp.sampler2}"
                                       for comp in dim_comparisons))

            # Initialize matrix
            p_matrix = np.ones((len(func_names), len(sampler_pairs)))

            for i, func in enumerate(func_names):
                for j, pair in enumerate(sampler_pairs):
                    for comp in dim_comparisons:
                        comp_pair = f"{comp.sampler1} vs {comp.sampler2}"
                        if comp.function_name == func and comp_pair == pair:
                            p_matrix[i, j] = comp.test_result.p_value
                            break

            # Plot heatmap
            im = ax.imshow(p_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=0.05)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('p-value')

            # Set ticks and labels
            ax.set_xticks(range(len(sampler_pairs)))
            ax.set_xticklabels(sampler_pairs, rotation=45, ha='right')
            ax.set_yticks(range(len(func_names)))
            ax.set_yticklabels(func_names)

            # Add text annotations
            for i in range(len(func_names)):
                for j in range(len(sampler_pairs)):
                    text = f'{p_matrix[i, j]:.3f}'
                    color = 'white' if p_matrix[i, j] < 0.025 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

            ax.set_title(f'Statistical Significance (p-values) - {dim}D')
            ax.set_xlabel('Sampler Comparisons')
            ax.set_ylabel('Functions')

        plt.tight_layout()

        if save_plot:
            filename = "statistical_significance.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved significance plot: {filepath}")

        plt.close()

    def plot_sampler_comparison_summary(self, results: List[RunResult],
                                        save_plot: bool = True) -> None:
        """Plot comprehensive summary of sampler performance."""

        framework = ExperimentFramework()
        summary_stats = framework.get_summary_statistics(results)

        if summary_stats.empty:
            print("No summary statistics to plot")
            return

        # Reset index to make it easier to work with
        summary_stats = summary_stats.reset_index()

        # Create subplot for different metrics
        metrics = ['best_fitness_mean', 'evaluations_mean', 'execution_time_mean', 'success_rate']
        metric_labels = ['Best Fitness (mean)', 'Evaluations (mean)', 'Execution Time (mean)', 'Success Rate']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            if metric not in summary_stats.columns:
                ax.text(0.5, 0.5, f'Metric {metric} not available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label)
                continue

            # Prepare data for plotting
            plot_data = []
            samplers = ['gaussian', 'sobol', 'halton']
            colors = ['blue', 'red', 'green']

            for sampler, color in zip(samplers, colors):
                sampler_data = summary_stats[summary_stats['sampler_type'] == sampler]
                if not sampler_data.empty:
                    values = sampler_data[metric].values
                    plot_data.append(values)

                    # Box plot
                    bp = ax.boxplot([values], positions=[len(plot_data) - 1],
                                    patch_artist=True, widths=0.6)
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_alpha(0.7)

            ax.set_xticks(range(len(samplers)))
            ax.set_xticklabels([s.title() for s in samplers])
            ax.set_ylabel(label)
            ax.set_title(f'Overall {label} Comparison')
            ax.grid(True, alpha=0.3)

            # Set log scale for appropriate metrics
            if metric in ['best_fitness_mean', 'evaluations_mean']:
                ax.set_yscale('log')

        plt.suptitle('Comprehensive Sampler Performance Summary', fontsize=16)
        plt.tight_layout()

        if save_plot:
            filename = "sampler_comparison_summary.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved summary plot: {filepath}")

        plt.close()

    def generate_all_visualizations(self, results: List[RunResult],
                                    analysis_results: Dict[str, Any] = None) -> None:
        """Generate all visualizations for the project."""

        print("Generating comprehensive visualizations...")
        print("=" * 50)

        # Get unique functions and dimensions
        framework = ExperimentFramework()
        df = framework.results_to_dataframe(results)

        if df.empty:
            print("No results to visualize")
            return

        functions = df['function_name'].unique()
        dimensions = sorted(df['dimension'].unique())

        print(f"Found {len(functions)} functions and {len(dimensions)} dimensions")

        # 1. Convergence curves (for selected functions)
        print("\n1. Generating convergence curves...")
        key_functions = ['sphere', 'rastrigin', 'ackley']  # Representative functions
        for func in key_functions:
            if func in functions:
                for dim in dimensions:
                    self.plot_convergence_curves(results, func, dim, save_plot=True)

        # 2. Performance boxplots
        print("\n2. Generating performance boxplots...")
        for metric in ['best_fitness', 'evaluations', 'execution_time']:
            self.plot_performance_boxplots(results, metric, save_plot=True)

        # 3. Performance heatmaps
        print("\n3. Generating performance heatmaps...")
        for metric in ['best_fitness', 'evaluations']:
            self.plot_performance_heatmap(results, metric, save_plot=True)

        # 4. Statistical significance (if available)
        if analysis_results and 'pairwise_comparisons' in analysis_results:
            print("\n4. Generating statistical significance plots...")
            self.plot_statistical_significance(analysis_results, save_plot=True)

        # 5. Overall summary
        print("\n5. Generating sampler comparison summary...")
        self.plot_sampler_comparison_summary(results, save_plot=True)

        print(f"\n✅ All visualizations saved to: {self.output_dir}")


def test_visualizations():
    """Test visualization generation with quick test data."""

    print("Testing Visualization Generation")
    print("=" * 50)

    try:
        # Load quick test results
        framework = ExperimentFramework()
        results = framework.load_results("quick_test_results.json")
        print(f"Loaded {len(results)} results from quick test")

        # Generate visualizations
        viz_gen = VisualizationGenerator()

        # Test individual plots
        print("\nTesting convergence curves...")
        viz_gen.plot_convergence_curves(results, 'sphere', 10, save_plot=True)

        print("\nTesting boxplots...")
        viz_gen.plot_performance_boxplots(results, 'best_fitness', save_plot=True)

        print("\nTesting heatmap...")
        viz_gen.plot_performance_heatmap(results, 'best_fitness', save_plot=True)

        print("\n✅ Visualization test completed successfully!")

    except FileNotFoundError:
        print("❌ No quick test results found. Run test_framework.py first.")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
