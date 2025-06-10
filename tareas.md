# Plan de Trabajo - Práctica Optativa CMA-ES
## Objetivo: Obtener la máxima puntuación (1.0 punto adicional)

---

## 📋 Resumen Ejecutivo

Para obtener la **máxima puntuación (1.0 punto)** según la rúbrica del profesor, necesitamos:
- **3 Procedimientos de Generación de Pseudonúmeros Aleatorios (PNA)**
- **12 Problemas de optimización**
- **Equipo de 4-5 integrantes** (requisito organizacional)
- **Fecha límite:** 9 de junio antes de las 23:59

⚠️ **CRÍTICO**: La puntuación depende tanto de cumplir la rúbrica (3 PNA × 12 problemas) como de la **calidad de la memoria y solidez de la experimentación**.

---

## 🎯 Requisitos Obligatorios

### 1. Implementación de 3 Variantes de Sampling (PNA) - OBLIGATORIAS
1. **Muestreo gaussiano clásico** (sampling tradicional de CMA-ES) ✅ OBLIGATORIO
2. **Secuencia de Sobol** (secuencia de baja discrepancia) ✅ OBLIGATORIO
3. **Técnica adicional de baja discrepancia** ✅ OBLIGATORIO - elegir una:
   - **Secuencia de Halton** (RECOMENDADO - más sencilla)
   - Secuencia de Hammersley  
   - Latin Hypercube Sampling

📝 **NOTA**: Estas 3 variantes son específicamente las que el profesor menciona como "variantes de sampling en CMA-ES", NO cualquier generador aleatorio.

### 2. Problemas de Optimización (12 total)
**Problemas SUGERIDOS por el profesor (mínimo 3, máximo recomendado 5):**
1. **Sphere** ✅ OBLIGATORIO (función más simple)
2. **Rosenbrock** ✅ OBLIGATORIO (valle estrecho)
3. **Rastrigin** ✅ OBLIGATORIO (multimodal)
4. **Ackley** ✅ RECOMENDADO (multimodal con ruido)
5. **Schwefel** ✅ RECOMENDADO (altamente multimodal)

**Problemas adicionales bien justificados (7 más para llegar a 12):**
6. **Griewank** (producto de cosenos)
7. **Levy** (multimodal difícil)
8. **Zakharov** (asimétrica)
9. **Michalewicz** (multimodal con restricciones)
10. **Beale** (valle estrecho 2D)
11. **Booth** (función cuadrática)
12. **Matyas** (función con interacción)

⚠️ **IMPORTANTE**: El profesor dice "pueden añadirse otros benchmarks **debidamente justificados**". Cada función adicional debe justificarse en el informe.

---

## 🔧 Tareas Técnicas a Implementar

### A. Algoritmo CMA-ES Base
- [x] Implementar CMA-ES estándar con muestreo gaussiano
- [x] Sistema de parámetros configurable (dimensión, población, etc.)
- [x] Criterios de convergencia
- [x] Logging y seguimiento de evolución

### B. Variantes de Sampling
- [x] **Gaussiano clásico**: Implementación base con numpy.random.multivariate_normal
- [ ] **Secuencia de Sobol**: Usar scipy.stats.qmc.Sobol + transformación Box-Muller
- [ ] **Tercera técnica**: Implementar Halton/Hammersley/LHS + transformación

### C. Funciones de Optimización
- [x] Implementar las 3 funciones básicas (Sphere, Rosenbrock, Rastrigin)
- [x] Definir dominios y dimensiones para cada función
- [x] Valores óptimos conocidos para cada problema
- [x] Funciones de evaluación eficientes
- [ ] Implementar las 9 funciones restantes

### D. Framework Experimental
- [x] Integración básica CMA-ES + funciones verificada
- [ ] Sistema de configuración experimental
- [ ] Múltiples ejecuciones con diferentes semillas
- [ ] Recolección de métricas:
  - Número de evaluaciones hasta convergencia
  - Mejor fitness encontrado
  - Tiempo de ejecución
  - Estadísticas de convergencia

---

## 📊 Análisis y Evaluación

### Métricas a Recopilar
- [ ] **Velocidad de convergencia**: Evaluaciones hasta alcanzar tolerancia
- [ ] **Robustez**: Desviación estándar entre ejecuciones
- [ ] **Coste computacional**: Tiempo de ejecución
- [ ] **Calidad de solución**: Distancia al óptimo global

### Análisis Estadístico
- [ ] Pruebas de normalidad (Shapiro-Wilk)
- [ ] Test de Wilcoxon signed-rank para comparaciones pareadas
- [ ] Test de Kruskal-Wallis para comparación múltiple
- [ ] Análisis de significancia estadística (p-value < 0.05)

### Visualizaciones
- [ ] Curvas de convergencia por función y método
- [ ] Boxplots comparativos de rendimiento
- [ ] Heatmaps de rendimiento por función/método
- [ ] Gráficos de distribución de poblaciones

---

## 📝 Entregables

### 1. Código Fuente
- [ ] **Estructura modular y documentada**
- [ ] **Instrucciones de reproducción** (README.md)
- [ ] **Dependencias claramente especificadas** (requirements.txt)
- [ ] **Scripts de ejecución automatizada**
- [ ] **Notebooks Jupyter para análisis**

### 2. Informe Técnico (~5 páginas) - ESTRUCTURA ESPECÍFICA DEL PROFESOR
- [ ] **Introducción**: Motivación y objetivos
- [ ] **Descripción de cada variante de sampling** ✅ OBLIGATORIO EXPLÍCITO
- [ ] **Configuración experimental** ✅ OBLIGATORIO EXPLÍCITO:
  - Dimensión utilizada
  - Tamaño de población  
  - Número de ejecuciones independientes
  - Semillas utilizadas
  - Criterios de parada
- [ ] **Resultados presentados con gráficos y tablas resumidas** ✅ OBLIGATORIO EXPLÍCITO
- [ ] **Contraste estadístico** ✅ OBLIGATORIO EXPLÍCITO (pruebas estadísticas)
- [ ] **Discusión crítica de los hallazgos** ✅ OBLIGATORIO EXPLÍCITO
- [ ] **Conclusiones y posibles extensiones del estudio** ✅ OBLIGATORIO EXPLÍCITO
- [ ] **Justificación adecuada de problemas tratados y generadores usados** ✅ OBLIGATORIO
- [ ] **Comparaciones/reflexiones entre procedimientos** ✅ OBLIGATORIO

---

## ⚙️ Configuración Experimental Sugerida

### Parámetros CMA-ES
- **Dimensión**: 10D, 20D (probar ambas)
- **Tamaño población**: λ = 4 + floor(3*ln(n)) donde n=dimensión
- **Número de ejecuciones**: 30 por configuración
- **Criterio parada**: 10^-8 tolerancia o 10^4*n evaluaciones máximo
- **Semillas**: Fijas y documentadas para reproducibilidad

### Hardware/Software
- **Lenguaje**: Python 3.8+
- **Librerías**: numpy, scipy, matplotlib, pandas, seaborn
- **Paralelización**: Multiprocessing para múltiples ejecuciones

---

## 📅 Cronograma de Trabajo

### Semana 1: Implementación Base
- Días 1-2: CMA-ES con muestreo gaussiano
- Días 3-4: Implementar 6 funciones benchmark
- Días 5-7: Framework experimental y testing

### Semana 2: Variantes de Sampling
- Días 1-3: Implementar Sobol y tercera técnica
- Días 4-5: Implementar 6 funciones restantes
- Días 6-7: Validación y debugging

### Semana 3: Experimentación
- Días 1-4: Ejecutar experimentos completos
- Días 5-7: Análisis estadístico y visualizaciones

### Semana 4: Informe
- Días 1-3: Redacción del informe técnico
- Días 4-5: Revisión y refinamiento
- Días 6-7: Preparación final y entrega

---

## ✅ Checklist Final Antes de Entrega

### Requisitos Técnicos para la Rúbrica (1.0 punto):
- [ ] **3 variantes de sampling CMA-ES** implementadas y validadas (gaussiano, Sobol, Halton)
- [ ] **12 problemas de optimización** funcionando correctamente
- [ ] **Experimentos completos** ejecutados (30+ runs por configuración)

### Requisitos de Calidad (factor multiplicador de puntuación):
- [ ] **Análisis riguroso** en velocidad convergencia, robustez y coste computacional ✅ CRÍTICO
- [ ] **Métricas adecuadas y pruebas estadísticas** implementadas ✅ CRÍTICO
- [ ] **Código fuente claramente documentado** con instrucciones reproducción ✅ CRÍTICO
- [ ] **Informe técnico ~5 páginas** con estructura específica del profesor ✅ CRÍTICO
- [ ] **Justificación de cada problema adicional** más allá de los 5 sugeridos ✅ CRÍTICO
- [ ] **Contraste estadístico riguroso** y discusión crítica ✅ CRÍTICO

### Entrega:
- [ ] **Equipo de 4-5 integrantes** formado
- [ ] **Archivo comprimido o repositorio** preparado
- [ ] **Entrega antes del 9 de junio a las 23:59** ✅ FECHA LÍMITE ESTRICTA

⚠️ **ADVERTENCIA**: Cumplir solo la rúbrica (3×12) no garantiza 1.0 punto. La puntuación final depende de la **calidad de la memoria y solidez de la experimentación**.

---

## 🏆 Resultado Esperado
**Puntuación objetivo: 1.0 punto adicional**

**Fórmula de puntuación del profesor:**
1. **Rúbrica base**: 3 PNA × 12 problemas = 1.0 punto máximo posible
2. **Factor de calidad**: Proporcional a la calidad de la memoria y solidez experimentación
3. **Puntuación final** = Rúbrica base × Factor de calidad

⚠️ **CRÍTICO**: Un trabajo que cumpla 3×12 pero con baja calidad puede obtener menos de 1.0 punto. La excelencia en experimentación y redacción es FUNDAMENTAL para obtener la máxima puntuación. 