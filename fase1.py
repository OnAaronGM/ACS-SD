import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from constants import CLUSTER_PERCENTILE, CLUSTER_MIN_THRESHOLD


# =========================================================
# Funciones métricas
# =========================================================

def support(mask: np.ndarray) -> float:
    return mask.mean()


def confidence(mask: np.ndarray, y: np.ndarray, target_value) -> float:
    return (y[mask] == target_value).mean()


def wracc(mask: np.ndarray, y: np.ndarray, target_value) -> float:
    """WRAcc = P(A∧T) - P(A)·P(T)."""
    coverage = mask.mean()
    conf_dataset = (y == target_value).mean()
    conf_subgroup = (y[mask] == target_value).mean()
    return coverage * (conf_subgroup - conf_dataset)


def lift(mask: np.ndarray, y: np.ndarray, target_value) -> float:
    """Lift = P(T|A) / P(T)."""
    p_t = (y == target_value).mean()
    if mask.sum() == 0 or p_t == 0:
        return 0.0
    return (y[mask] == target_value).mean() / p_t


def mutual_information(mask: np.ndarray, y: np.ndarray) -> float:
    """MI binaria entre condición y target."""
    return mutual_info_score(mask.astype(int), y)


# =========================================================
# Cálculo principal de estadísticas por condición
# =========================================================

def compute_triplet_stats(
    df: pd.DataFrame, target_col: str, target_value=1
) -> pd.DataFrame:
    """
    Para cada par (feature, value) calcula support, wracc, lift y confidence.
    Filtra condiciones con soporte mínimo (√n) o lift < 1.
    """
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    min_support = math.sqrt(len(df))

    stats = []
    for col in X.columns:
        for v in X[col].unique():
            mask = X[col].values == v

            if mask.sum() < min_support:
                continue

            l = lift(mask, y, target_value)
            if l < 1:
                continue

            stats.append((
                f"{col}:{v}",
                support(mask),
                wracc(mask, y, target_value),
                l,
                confidence(mask, y, target_value),
            ))

    return pd.DataFrame(stats, columns=["condition", "support", "wracc", "lift", "conf"])


# =========================================================
# Frente de Pareto
# =========================================================

def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Devuelve las soluciones no dominadas (maximización en ambos ejes)."""
    data = df[[x_col, y_col]].values
    n = len(data)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates = (
            (data[:, 0] >= data[i, 0])
            & (data[:, 1] >= data[i, 1])
            & ((data[:, 0] > data[i, 0]) | (data[:, 1] > data[i, 1]))
        )
        dominates[i] = False
        if dominates.any():
            is_pareto[i] = False

    return df[is_pareto].copy()


def plot_pareto(
    stats_df: pd.DataFrame, name: str | None = None, plot: bool = True
) -> pd.DataFrame:
    """Calcula y opcionalmente dibuja el frente de Pareto (support vs conf)."""
    pareto_df = pareto_front(stats_df, "support", "conf")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(stats_df["support"], stats_df["conf"], s=10, label="Todas")
        ax.scatter(pareto_df["support"], pareto_df["conf"], s=40, label="Pareto")

        pareto_sorted = pareto_df.sort_values("support")
        ax.plot(pareto_sorted["support"], pareto_sorted["conf"])

        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Pareto front (Support vs Confidence)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"./plots/pareto_{name}.png", dpi=300)
        plt.close(fig)

    return pareto_df


# =========================================================
# Similitud de subgrupos (Jaccard)
# =========================================================

def jaccard_matrix(
    stats_df: pd.DataFrame, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la matriz de similitud de Jaccard entre todos los subgrupos.

    Optimización ②: en lugar de un doble loop Python O(n² · instancias),
    se apilan las máscaras en una matriz booleana M (n × instancias) y se
    calculan todas las intersecciones/uniones con una sola operación BLAS.
      intersección[i,j] = (M[i] & M[j]).sum()  →  M_int @ M_int.T
      unión[i,j]        = IS[i] + IS[j] − intersección[i,j]

    Parameters
    ----------
    stats_df : DataFrame con columna 'condition' en formato 'feature:value'
    X        : DataFrame original (sin target) con las mismas filas

    Returns
    -------
    (matriz nxn de Jaccard, array de nombres de condición)
    """
    conditions = stats_df["condition"].values

    masks = np.stack([
        X[col].values == val
        for cond in conditions
        for col, val in [cond.split(":")]
    ])                                                    # (n, n_instances) bool

    M = masks.astype(np.int32)
    intersection = M @ M.T                               # (n, n) int  — una op BLAS

    IS = masks.sum(axis=1)                               # (n,) int
    union = IS[:, None] + IS[None, :] - intersection    # (n, n) int

    jaccard_mat = np.where(union > 0, intersection / union, 0.0)
    return jaccard_mat.astype(float), conditions


# =========================================================
# Clustering jerárquico
# =========================================================

def _compute_adaptive_threshold(D_matrix: np.ndarray) -> float:
    """Umbral adaptativo basado en densidades locales de k vecinos más cercanos."""
    k = min(5, D_matrix.shape[0] - 1)
    local_distances = []
    for i in range(D_matrix.shape[0]):
        sorted_dists = np.sort(D_matrix[i, :])
        local_distances.append(np.mean(sorted_dists[1 : k + 1]))

    threshold = np.percentile(local_distances, CLUSTER_PERCENTILE)
    threshold = max(threshold, CLUSTER_MIN_THRESHOLD)
    print(f"Threshold adaptativo basado en densidad: {threshold:.3f}")
    return threshold


def hierarchical_clustering(
    W: np.ndarray,
    actions,
    names: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    method: str = "average",
    plot: bool = False,
    figsize: tuple = (8, 4),
    rotation: int = 90,
) -> dict:
    """
    Clustering jerárquico basado en redundancia (similitud → distancia).

    Parameters
    ----------
    W       : Matriz nxn de similitud (e.g. Jaccard)
    actions : Objeto ActionSpace con método translate()
    names   : Etiquetas de subgrupos
    quality : Calidad de cada subgrupo (para elegir representante)
    method  : Método de linkage ('average' recomendado)
    plot    : Dibujar dendrograma

    Returns
    -------
    Diccionario {cluster_id: {'indices', 'n_ants', 'representative'}}
    """
    n = len(W)
    if names is None:
        names = np.array([str(i) for i in range(n)])

    D = 1 - W
    D_condensed = squareform(D)
    D_matrix = squareform(D_condensed)

    threshold = _compute_adaptive_threshold(D_matrix)

    Z = linkage(D_condensed, method=method)
    clusters = fcluster(Z, t=threshold, criterion="distance")

    cluster_dict = {}
    for cluster_id in np.unique(clusters):
        idx = np.where(clusters == cluster_id)[0]

        intra_dist = (
            np.mean([D_matrix[i, j] for i in idx for j in idx if i < j])
            if len(idx) > 1
            else 0.0
        )
        n_ants = max(2, int(np.sqrt(len(idx)) * (1 + intra_dist)))

        if len(idx) == 1:
            col, val = names[idx[0]].split(":")
            representative = actions.translate(col, val)
        else:
            best_idx = idx[np.argmax(quality[idx])]
            col, val = names[best_idx].split(":")
            representative = actions.translate(col, val)

        cluster_list = []
        for value in idx:
            col, val = names[value].split(":")
            cluster_list.append(actions.translate(col, val))

        cluster_dict[cluster_id] = {
            "indices": cluster_list,
            "n_ants": n_ants,
            "representative": representative,
        }

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(Z, labels=names, leaf_rotation=rotation, leaf_font_size=10,
                   color_threshold=None, ax=ax)
        ax.axhline(y=threshold, linestyle="--", alpha=0.3)
        ax.set_ylabel("Distance = 1 - redundancy")
        ax.set_title("Hierarchical clustering (redundancy-based)")
        fig.tight_layout()
        plt.show()

    return cluster_dict


# =========================================================
# Punto de entrada
# =========================================================

def main(data: pd.DataFrame, target: list, actions, cluster: bool = False) -> dict:
    """
    Ejecuta fase 1: calcula condiciones candidatas y opcionalmente las agrupa.

    Returns
    -------
    Diccionario de clusters con sus semillas iniciales para ACS.
    """
    stats_df = compute_triplet_stats(data, target_col=target[0], target_value=target[1])
    pareto_df = plot_pareto(stats_df, name=None, plot=False)
    #return pareto_df
    if cluster and len(pareto_df) > 1:
        jaccard_mat, condition_names = jaccard_matrix(pareto_df, data)
        return hierarchical_clustering(
            jaccard_mat, actions, condition_names, stats_df["wracc"].values, plot=False
        )

    col, val = pareto_df["condition"].iloc[0].split(":")
    sg = actions.translate(col, val)
    return {1: {"indices": [sg], "n_ants": 2, "representative": sg}}