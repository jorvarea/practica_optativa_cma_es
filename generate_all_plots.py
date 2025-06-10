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

    print("Proyecto CMA-ES - Generación Completa de Visualizaciones")
    print("-" * 60)

    # Check if we have results to work with
    results_file = "quick_test_results.json"
    if not Path("results", results_file).exists():
        print(f"Archivo de resultados results/{results_file} no encontrado.")
        print("Por favor ejecuta test_framework.py primero para generar datos de prueba.")
        return

    # Load experimental results
    print("Cargando resultados experimentales...")
    framework = ExperimentFramework()
    results = framework.load_results(results_file)
    print(f"Cargados {len(results)} resultados experimentales")

    # Perform statistical analysis
    print("Realizando análisis estadístico...")
    analyzer = StatisticalAnalyzer()
    analysis_results = analyzer.comprehensive_analysis(results)

    print(f"Análisis estadístico completado:")
    print(f"  - Tests de normalidad: {len(analysis_results['normality_tests'])}")
    print(f"  - Comparaciones pareadas: {len(analysis_results['pairwise_comparisons'])}")
    if 'multiple_comparisons' in analysis_results:
        print(f"  - Comparaciones múltiples: {len(analysis_results['multiple_comparisons'])}")

    # Generate visualizations
    print("Generando visualizaciones completas...")
    viz_gen = VisualizationGenerator(output_dir="plots")

    # Generate all plots
    viz_gen.generate_all_visualizations(results, analysis_results)

    # Summary
    print("-" * 60)
    print("GENERACIÓN DE VISUALIZACIONES COMPLETADA EXITOSAMENTE!")
    print(f"Todos los gráficos guardados en: plots/")
    print("Visualizaciones generadas:")
    print("  1. Curvas de convergencia para funciones clave (Sphere, Rastrigin, Ackley)")
    print("  2. Boxplots de rendimiento (Mejor Fitness, Evaluaciones, Tiempo de Ejecución)")
    print("  3. Mapas de calor de rendimiento (Mejor Fitness, Evaluaciones)")
    print("  4. Visualización de significancia estadística")
    print("  5. Resumen completo de comparación de muestreadores")

    # List all generated files
    plots_dir = Path("plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"Total de gráficos generados: {len(plot_files)}")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file.name}")

    print("Estas visualizaciones están listas para incluir en tu reporte técnico!")
    print("   Cada gráfico está guardado a 300 DPI para impresión de alta calidad.")


if __name__ == "__main__":
    main()
