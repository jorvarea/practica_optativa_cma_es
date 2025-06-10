#!/usr/bin/env python3
"""
Main script to generate comprehensive visualizations for CMA-ES project.
This script generates all required plots for the technical report.
"""

import os
import sys
from pathlib import Path

sys.path.append('src')

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for batch processing

from experiments import ExperimentFramework
from statistical_analysis import StatisticalAnalyzer
from visualizations import VisualizationGenerator


def main():
    """Generate all visualizations for the CMA-ES project."""

    print("🎨 CMA-ES Project - Comprehensive Visualization Generation")
    print("=" * 60)

    # Check if we have results to work with
    results_file = "quick_test_results.json"
    if not Path("results", results_file).exists():
        print(f"❌ Results file results/{results_file} not found.")
        print("Please run test_framework.py first to generate test data.")
        return

    try:
        # Load experimental results
        print("📊 Loading experimental results...")
        framework = ExperimentFramework()
        results = framework.load_results(results_file)
        print(f"✅ Loaded {len(results)} experimental results")

        # Perform statistical analysis
        print("\n📈 Performing statistical analysis...")
        analyzer = StatisticalAnalyzer()
        analysis_results = analyzer.comprehensive_analysis(results)

        print(f"✅ Statistical analysis completed:")
        print(f"  - Normality tests: {len(analysis_results['normality_tests'])}")
        print(f"  - Pairwise comparisons: {len(analysis_results['pairwise_comparisons'])}")
        if 'multiple_comparisons' in analysis_results:
            print(f"  - Multiple comparisons: {len(analysis_results['multiple_comparisons'])}")

        # Generate visualizations
        print("\n🎨 Generating comprehensive visualizations...")
        viz_gen = VisualizationGenerator(output_dir="plots")

        # Generate all plots
        viz_gen.generate_all_visualizations(results, analysis_results)

        # Summary
        print("\n" + "=" * 60)
        print("✅ VISUALIZATION GENERATION COMPLETED SUCCESSFULLY!")
        print(f"📁 All plots saved to: plots/")
        print("\n📋 Generated visualizations:")
        print("  1. Convergence curves for key functions (Sphere, Rastrigin, Ackley)")
        print("  2. Performance boxplots (Best Fitness, Evaluations, Execution Time)")
        print("  3. Performance heatmaps (Best Fitness, Evaluations)")
        print("  4. Statistical significance visualization")
        print("  5. Comprehensive sampler comparison summary")

        # List all generated files
        plots_dir = Path("plots")
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            print(f"\n📄 Total plots generated: {len(plot_files)}")
            for plot_file in sorted(plot_files):
                print(f"  - {plot_file.name}")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎯 These visualizations are ready for inclusion in your technical report!")
    print("   Each plot is saved at 300 DPI for high-quality printing.")


if __name__ == "__main__":
    main()
