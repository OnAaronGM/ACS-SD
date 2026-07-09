# =========================================================
# Constantes globales del algoritmo ACS-SD
# =========================================================

# --- ACS hyperparámetros ---
ACS_Q0: float = 0.85               # Umbral greedy vs exploratorio
ACS_ALPHA: float = 1.0             # Exponente feromona
ACS_BETA: float = 2.0              # Exponente heurística
ACS_RHO: float = 0.1               # Tasa de evaporación
ACS_DEPOSIT_FACTOR: float = 0.1    # Factor de depósito de feromona
ACS_PHEROMONE_SCALE: float = 10.0  # Multiplicador de calidad en depósito global
RANDOM_SEED_NUMBER = 200           # Valor de semilla aleatoria

# --- Criterios de parada ---
EARLY_STOP_NO_IMPROVE: int = 2     # Iteraciones sin mejora antes de parar
MIN_ENTROPY_THRESHOLD: float = 0.1 # Umbral mínimo de entropía para continuar

# --- Clustering jerárquico (fase1) ---
CLUSTER_PERCENTILE: int = 40       # Percentil para umbral adaptativo de densidad
CLUSTER_MIN_THRESHOLD: float = 0.1 # Umbral mínimo de distancia en clustering
