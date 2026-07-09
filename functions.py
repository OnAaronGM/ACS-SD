import copy
import math

import numpy as np
import pandas as pd
from scipy.stats import entropy


# -------------------------
# WRAcc metric
# -------------------------

def wracc_measure(ID: int, PD: int, IS: int, PS: int) -> float:
    """WRAcc = P(A) * (P(T|A) - P(T))."""
    if PS == 0:
        return -1.0
    coverage = IS / ID
    wracc = coverage * ((PS / IS) - (PD / ID))
    return wracc


# -------------------------
# OR metric
# -------------------------

def OR_measure(
    ID: int, PD: int, IS: int, PS: int
) -> tuple[float | None, tuple[float, float] | None]:
    """Odds Ratio con intervalo de confianza."""
    if PS == 0:
        return None, None

    a = PS
    b = IS - PS
    c = PD - PS
    d = (ID - PD) - b

    # Corrección de continuidad de Haldane-Anscombe
    if b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    w = 1.39 * math.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    OR = (a * d) / (b * c)
    OR_interval = (OR * math.exp(-w), OR * math.exp(+w))
    score = math.log(OR) - w
    return OR, OR_interval, score


def odd_equivalent(OR: float) -> int:
    """Clasifica el OR en 4 categorías ordinales (umbral de Cohen)."""
    if OR < 1.68:
        return 1
    elif OR < 3.47:
        return 2
    elif OR < 6.71:
        return 3
    return 4


# -------------------------
# Information Gain
# -------------------------

def calculate_info_gained(ID: int, PD: int, IS: int, PS: int) -> float:
    """Ganancia de información normalizada al dividir un nodo (ID, PD) en (IS, PS)."""
    p1 = PD / ID
    H1 = entropy([p1, 1 - p1], base=2)

    if H1 == 0:  # Nodo padre ya es 100 % puro
        return 0.0

    remainder = ID - IS
    p_complement = (PD - PS) / remainder if remainder != 0 else 0.0

    x_in = IS / ID
    x_out = remainder / ID

    gain = (
        H1
        - x_in * entropy([PS / IS, 1 - PS / IS], base=2)
        - x_out * entropy([p_complement, 1 - p_complement], base=2)
    ) / H1

    return 0.0 if np.isnan(gain) else gain


# -------------------------
# Reglas / máscaras
# -------------------------

def apply_action(rule: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Aplica una acción sobre una regla"""
    rule_copy = rule.copy()
    mask = action != 0
    rule_copy[mask] = action[mask]
    return rule_copy


def rule_to_index_mask(rule: np.ndarray, sdt) -> np.ndarray:
    """Devuelve una máscara booleana de las instancias cubiertas por una regla."""
    if not np.any(rule):
        return np.ones(len(sdt.data), dtype=bool)

    mask = np.ones(len(sdt.data), dtype=bool)
    for index, value in enumerate(rule):
        if value != 0:
            attr = sdt.categorical_cols[index]
            val = sdt.map_attrs_vals[attr][value - 1]
            mask &= sdt.data[attr] == val
    return mask


# -------------------------
# Poda anticipada
# -------------------------

def evaluate_continue_path(
    ID: int, PD: int, sg, best_quality: float, actions
) -> bool:
    """
    Cotas de poda: devuelve False si el subgrupo no puede superar best_quality.

    Aplica dos cotas (simple y geométrica) sobre IS y PS.
    """
    IS = sg.IS
    PS = sg.PS
    alpha = PD / ID

    s_min = (best_quality * ID) / (1 - alpha)
    if IS < s_min:
        return False

    p_min = (PS * alpha) + (best_quality * ID)
    if PS < p_min:
        return False

    return True


# -------------------------
# Generación de estadísticas
# -------------------------

def generate_df_custom(
    results: list,
    PD_map: dict,
    ID_base: int,
    og_task_param,
) -> tuple[list, np.ndarray]:
    """
    Calcula las estadísticas de una lista de subgrupos descubiertos.

    Parameters
    ----------
    results      : lista de (Pattern, target_value)
    PD_map       : {target_value: nº positivos en dataset completo}
    ID_base      : tamaño total del dataset
    og_task_param: SubgroupDiscoveryTask original (se clona internamente)

    Returns
    -------
    (lista de métricas agregadas, máscara de instancias cubiertas)
    """
    og_task = copy.deepcopy(og_task_param)
    PD_map_bubble = og_task.get_map_positives()
    print("GENERANDO STATS")

    stats: dict[str, list] = {
        "length": [], "wracc_list": [], "coverage_list": [], "OR_list": [],
        "wracc_set": [], "coverage_set": [], "OR_set": [],
        "confidence": [], "IS": [], "PS": [], "target": [],
    }

    for sg, target in results:

        stats["length"].append(np.count_nonzero(sg.state))

        quality_global = wracc_measure(ID_base, PD_map[target], sg.IS, sg.PS)
        quality_bubble = wracc_measure(og_task.ID, PD_map_bubble[target], sg.IS, sg.PS)
        stats["wracc_list"].append(quality_global)
        stats["wracc_set"].append(quality_bubble)

        stats["coverage_list"].append(sg.IS / ID_base)
        stats["coverage_set"].append(sg.IS / og_task.ID)

        stats["confidence"].append(sg.PS / sg.IS)

        OR, _ , _= OR_measure(ID_base, PD_map[target], sg.IS, sg.PS)
        stats["OR_list"].append(odd_equivalent(OR))
        OR, _, _ = OR_measure(og_task.ID, PD_map_bubble[target], sg.IS, sg.PS)
        stats["OR_set"].append(odd_equivalent(OR))

        stats["IS"].append(sg.IS)
        stats["PS"].append(sg.PS)
        stats["target"].append(target)

        og_task.data = og_task.data.loc[
            ~og_task.data.index.isin(og_task.data[sg.instances_covered].index)
        ]
        og_task.calculate_constants()
        PD_map_bubble = og_task.get_map_positives()

    df = pd.DataFrame(data=stats)
    summary = [
        len(results),
        np.mean(df["length"]),
        np.sum(df["wracc_list"]),
        np.mean(df["wracc_list"]),
        np.sum(df["coverage_list"]),
        np.mean(df["coverage_list"]),
        np.mean(df["OR_list"]),
        np.mean(df["wracc_set"]),
        np.mean(df["coverage_set"]),
        np.mean(df["OR_set"]),
        np.mean(df["confidence"]),
        np.sum(df["PS"]) / np.sum(df["IS"]),
    ]
    return summary
