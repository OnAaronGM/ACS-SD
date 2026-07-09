import copy
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy

import fase1
import functions
from constants import (
    ACS_Q0,
    ACS_ALPHA,
    ACS_BETA,
    ACS_RHO,
    ACS_DEPOSIT_FACTOR,
    ACS_PHEROMONE_SCALE,
    EARLY_STOP_NO_IMPROVE,
    MIN_ENTROPY_THRESHOLD,
)

""" np.random.seed(200)
random.seed(200) """


# =========================================================
# SubgroupDiscoveryTask
# =========================================================

class SubgroupDiscoveryTask:
    """Encapsula todos los parámetros necesarios para subgroup discovery."""

    def init_params(
        self,
        data: pd.DataFrame,
        categorical_cols: list[str],
        target: list,
        depth: int = 3,
        timeout: int = 3600,
    ) -> None:
        self.data = data
        self.target = target.copy()  # [variable, valor]
        self.depth = depth
        self.name_df = ""
        self.timeout = timeout
        self.categorical_cols = categorical_cols
        self.map_attrs_vals: dict = {}
        self.n_iters = 10
        self.ants_per_iter = 10
        self.max_steps_per_ant = len(self.categorical_cols)

    def calculate_constants(self) -> None:
        self.mask_positives = self.data[self.target[0]] == self.target[1]
        self.PD = int(self.mask_positives.sum())
        self.ID = len(self.data)

    def get_map_positives(self) -> dict:
        return {
            value: int((self.data[self.target[0]] == value).sum())
            for value in self.data[self.target[0]].unique()
        }


# =========================================================
# Pattern
# =========================================================

class Pattern:
    """Representa un subgrupo (regla) junto con sus métricas."""

    def __init__(self, rule: np.ndarray, target=None) -> None:
        self.selectors: list = []
        self.state = rule
        self.instances_covered = functions.rule_to_index_mask(rule, ctx.sdt)
        self.IS = int(self.instances_covered.sum())
        self.target = target
        self.PS = int((self.instances_covered & ctx.sdt.mask_positives).sum())
        self.wracc = 0.0
        self.coverage = 0.0
        self.confidence = 0.0
        self.OR = 0.0
        self.OR_interval: tuple[float, float] = (0.0, 0.0)
        self.score = 0
        self.IG = 0.0

    def getAttrs(self) -> list:
        return [selector[0] for selector in self.selectors]

    def getVals(self) -> list:
        return [selector[1] for selector in self.selectors]

    def getTriplets(self) -> list[tuple]:
        return [(selector[0], selector[1]) for selector in self.selectors]

    def setTriplets(self) -> None:
        indices = np.nonzero(self.state)[0]
        for idx, value in zip(indices, self.state[indices]):
            col = ctx.sdt.categorical_cols[idx]
            triplet = (ctx.sdt.map_attrs_vals[col], ctx.sdt.map_attrs_vals[col][value - 1])
            self.selectors.append(triplet)

    def setValues(self, IP: int, PP: int) -> None:
        self.wracc = functions.wracc_measure(self.IS, self.PS, IP, PP)
        self.OR, self.OR_interval = functions.OR_measure(
            self.IS, self.PS, ctx.sdt.ID, ctx.sdt.mask_positives[self.target][1]
        )

    @classmethod
    def from_mask(cls, rule: np.ndarray, mask: np.ndarray, target=None) -> "Pattern":
        """
        Constructor rápido que reutiliza una máscara ya calculada.
        Evita llamar a rule_to_index_mask (loop Python) en el hot path.
        """
        obj = object.__new__(cls)
        obj.selectors = []
        obj.state = rule
        obj.instances_covered = mask
        obj.IS = int(mask.sum())
        obj.target = target
        obj.PS = int((mask & ctx.sdt.mask_positives).sum())
        obj.wracc = 0.0
        obj.coverage = 0.0
        obj.confidence = 0.0
        obj.OR = 0.0
        obj.OR_interval = (0.0, 0.0)
        obj.IG = 0.0
        return obj

    def __repr__(self) -> str:
        return " AND ".join(f"{s[0]}=={s[1]}" for s in self.selectors)


# =========================================================
# States
# =========================================================

class States:
    """Registro de estados visitados (caché de IDs)."""

    def init_params(self) -> None:
        self.states: dict = defaultdict(int)

    def check_state(self, state: tuple) -> bool:
        return state in self.states

    def add_state(self, state: tuple) -> None:
        if state not in self.states:
            self.states[state] = len(self.states) + 1

    def get_state_id(self, state: np.ndarray) -> int:
        key = tuple(state)
        if not self.check_state(key):
            self.add_state(key)
        return self.states[key]


# =========================================================
# ActionSpace
# =========================================================

class ActionSpace:
    """Gestiona el espacio de acciones (pares atributo-valor)."""

    def init_params(self) -> None:
        self.actions: list[np.ndarray] = []
        for index_attr, col in enumerate(ctx.sdt.categorical_cols):
            vals = sorted(ctx.sdt.data[col].unique().tolist())
            ctx.sdt.map_attrs_vals[col] = vals
            for index, _ in enumerate(vals):
                action = np.zeros(len(ctx.sdt.categorical_cols), dtype=int)
                action[index_attr] = index + 1
                self.actions.append(action)

        self.n_actions = len(self.actions)
        self.action_to_id = {tuple(a): i for i, a in enumerate(self.actions)}
        self.id_to_action = dict(enumerate(self.actions))

        # Matriz (n_acciones × n_attrs) para vectorizar valid_actions_mask.
        self.actions_matrix = np.stack(self.actions)  # (n_actions, n_attrs) int

        # Matriz de máscaras booleanas (n_acciones × n_instancias).
        self._compute_masks()

    def _compute_masks(self) -> None:
        """
        Pre-computa una máscara booleana por acción sobre ctx.sdt.data actual.
        Debe llamarse cada vez que ctx.sdt.data cambia (update_action_space).
        Cada acción tiene exactamente un atributo no nulo, así que la máscara
        es simplemente data[col] == val para ese par.
        """
        n_inst = len(ctx.sdt.data)
        self.masks_matrix = np.empty((self.n_actions, n_inst), dtype=bool)
        for i, action in enumerate(self.actions):
            attr_idx = int(np.flatnonzero(action)[0])          # único no-cero
            val_idx = int(action[attr_idx])
            col = ctx.sdt.categorical_cols[attr_idx]
            val = ctx.sdt.map_attrs_vals[col][val_idx - 1]
            self.masks_matrix[i] = ctx.sdt.data[col].values == val

    def translate(self, col: str, value) -> np.ndarray:
        """Convierte un par (columna, valor) en vector de acción."""
        index_col = ctx.sdt.categorical_cols.index(col)
        index_value = ctx.sdt.map_attrs_vals[col].index(value) + 1
        action = np.zeros(len(ctx.sdt.categorical_cols), dtype=int)
        action[index_col] = index_value
        return action

    def update_action_space(self) -> np.ndarray:
        """
        Invalida acciones cuyo valor ya no existe en los datos actuales
        y refresca masks_matrix para que refleje el subconjunto de datos vigente.
        """
        valid_actions = np.ones(self.n_actions, dtype=int)
        for index_attr, col in enumerate(ctx.sdt.categorical_cols):
            vals = ctx.sdt.data[col].unique().tolist()
            removed = ~np.isin(np.array(ctx.sdt.map_attrs_vals[col]), np.array(vals))
            for index in np.where(removed)[0]:
                action = np.zeros(len(ctx.sdt.categorical_cols), dtype=int)
                action[index_attr] = index + 1
                valid_actions[self.action_to_id[tuple(action)]] = 0

        # Las máscaras apuntan a las filas del DataFrame actual; hay que regenerarlas.
        self._compute_masks()
        return valid_actions

    def get_action_id(self, action: np.ndarray) -> int:
        return self.action_to_id[tuple(action)]

    def valid_actions_mask(self, current_rule: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """
        Máscara de acciones válidas: una acción es válida si extiende la regla
        en al menos un atributo aún no fijado.
        """
        free = current_rule == 0                                       # (n_attrs,)
        can_extend = np.any((self.actions_matrix != 0) & free, axis=1) # (n_actions,)
        return can_extend.astype(int) & seeds


# =========================================================
# ACS (feromona + heurística)
# =========================================================

class ACS:
    """Colonia de hormigas con tres tipos de feromona (single, pair, path)."""

    def init_params(
        self,
        value: float,
        alpha: float = ACS_ALPHA,
        beta: float = ACS_BETA,
        rho: float = ACS_RHO,
        deposit_factor: float = ACS_DEPOSIT_FACTOR,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.deposit_factor = deposit_factor
        self.feromonas_single = defaultdict(lambda: value)
        self.feromonas_pair = defaultdict(lambda: value)
        self.feromonas_path = defaultdict(lambda: value)

    def reset(self, value: float) -> None:
        self.feromonas_single = defaultdict(lambda: value)
        self.feromonas_pair = defaultdict(lambda: value)
        self.feromonas_path = defaultdict(lambda: value)

    def acs_transition_probs(
        self,
        sg: Pattern,
        valid_actions: np.ndarray,
        path: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_actions_update = valid_actions.copy()
        feromonas = np.zeros_like(valid_actions, dtype=float)
        heuristicas = np.zeros_like(valid_actions, dtype=float)
        new_subgroups = np.zeros_like(valid_actions, dtype=object)

        for index in np.flatnonzero(valid_actions):
            action = ctx.actions.id_to_action[index]
            new_rule = sg.state | action
            
            # Añadir una condición equivale a intersectar (AND) las máscaras.
            new_mask = sg.instances_covered & ctx.actions.masks_matrix[index]
            new_sg = Pattern.from_mask(new_rule, new_mask, sg.target)
            # ──────────────────────────────────────────────────────────────

            if new_sg.IS == 0 or new_sg.PS == 0:
                valid_actions_update[index] = 0
                continue

            f1 = self.feromonas_single[ctx.states.get_state_id(action)]
            f2 = self.feromonas_pair[ctx.states.get_state_id(path[-1] | action)]
            f3 = self.feromonas_path[ctx.states.get_state_id(new_rule)]
            feromonas[index] = f1 + f2 + f3

            heuristicas[index] = functions.calculate_info_gained(
                sg.IS, sg.PS, new_sg.IS, new_sg.PS
            )
            new_subgroups[index] = new_sg

        if np.any(heuristicas):
            scores = np.power(feromonas, self.alpha) * np.power(heuristicas, self.beta)
            return scores, new_subgroups, valid_actions_update

        return heuristicas, new_subgroups, valid_actions_update

    def update_feromonas_local(
        self, subgroup: Pattern, path: list, action: np.ndarray
    ) -> None:
        decay = 1 - self.rho
        self.feromonas_single[ctx.states.get_state_id(action)] *= decay
        self.feromonas_pair[ctx.states.get_state_id(path[-1] | action)] *= decay
        self.feromonas_path[ctx.states.get_state_id(subgroup.state)] *= decay

    def update_feromonas_global(self, best_path: list, best_wracc: float) -> None:
        deposit = self.deposit_factor * ACS_PHEROMONE_SCALE * best_wracc
        decay = 1 - self.rho
        for index, state in enumerate(best_path):
            state_id = ctx.states.get_state_id(state)
            self.feromonas_single[state_id] = decay * self.feromonas_single[state_id] + deposit
            if index > 0:
                pair_id = ctx.states.get_state_id(best_path[index - 1] | best_path[index])
                self.feromonas_pair[pair_id] = decay * self.feromonas_pair[pair_id] + deposit
                path_id = ctx.states.get_state_id(np.maximum.reduce(best_path[: index + 1]))
                self.feromonas_path[path_id] = decay * self.feromonas_path[path_id] + deposit


# =========================================================
# Context (singleton global)
# =========================================================

class Context:
    def __init__(self) -> None:
        self.sdt = SubgroupDiscoveryTask()
        self.states = States()
        self.actions = ActionSpace()
        self.acs = ACS()

    def reset(self) -> None:
        self.sdt = SubgroupDiscoveryTask()
        self.states = States()
        self.actions = ActionSpace()
        self.acs = ACS()


ctx = Context()


# =========================================================
# Helpers del runner
# =========================================================

def evaluate_seeds(
    seeds: np.ndarray, actions: dict, sqrt_ID: float
) -> tuple[np.ndarray, float]:
    """
    Elimina semillas inválidas y devuelve la feromona inicial (WRAcc máximo).

    Optimización ①: calcula IS y PS para todos los seeds activos de una vez
    con operaciones matriciales sobre masks_matrix en lugar de crear un Pattern
    por cada semilla.
    """
    active = np.flatnonzero(seeds)
    if len(active) == 0:
        return seeds, 0.0

    active_masks = ctx.actions.masks_matrix[active]          # (n_active, n_inst)
    IS_arr = active_masks.sum(axis=1).astype(float)          # (n_active,)
    PS_arr = (active_masks & ctx.sdt.mask_positives.values).sum(axis=1).astype(float)

    alpha = ctx.sdt.PD / ctx.sdt.ID
    # Tres condiciones de filtrado vectorizadas
    valid = (
        (IS_arr >= sqrt_ID)
        & (PS_arr > 0)
        & (PS_arr * ctx.sdt.ID / ctx.sdt.PD >= sqrt_ID)
    )
    seeds[active[~valid]] = 0

    valid_IS = IS_arr[valid]
    valid_PS = PS_arr[valid]
    if len(valid_IS) == 0:
        return seeds, 0.0

    wr_arr = (valid_IS / ctx.sdt.ID) * (valid_PS / valid_IS - alpha)
    return seeds, float(wr_arr.max())


def _select_action(probs: np.ndarray) -> int:
    """Política ε-greedy: greedy con prob q0, aleatorio en caso contrario."""
    return int(np.argmax(probs)) if np.random.rand() <= ACS_Q0 else np.random.choice(len(probs), p=probs)


def _run_ant(
    info: dict,
    ant_idx: int,
    seeds: np.ndarray,
    actions: dict,
    best_path: list,
    ID_base: int,
    PD_map: dict,
) -> tuple[Pattern | None, list, list[float]]:
    """Ejecuta un único agente hormiga y devuelve (subgrupo, camino, calidades)."""
    #rule = random.choice(info)
    rule = info["representative"] if ant_idx % 2 == 0 else random.choice(info["indices"])
    sg = Pattern(rule)
    sg.OR, sg.OR_interval, sg.score = functions.OR_measure(ctx.sdt.ID, ctx.sdt.PD, sg.IS, sg.PS)
    ant_path = [sg.state]

    for _ in range(1, ctx.sdt.max_steps_per_ant):
        valid = ctx.actions.valid_actions_mask(sg.state, seeds)
        if not valid.any():
            break

        probs_raw, subgroups, valid = ctx.acs.acs_transition_probs(sg, valid, ant_path)
        if not probs_raw.any():
            break

        probs = probs_raw / probs_raw.sum()
        action_id = _select_action(probs)
        subgroup_chosen = subgroups[action_id]

        ctx.acs.update_feromonas_local(
            subgroup=subgroup_chosen, path=ant_path, action=actions[action_id]
        )

        # Condición de parada: sin mejora de OR
        subgroup_chosen.OR, subgroup_chosen.OR_interval, subgroup_chosen.score = functions.OR_measure(
            ctx.sdt.ID, ctx.sdt.PD, subgroup_chosen.IS, subgroup_chosen.PS
        )
        if subgroup_chosen.OR_interval[0] < sg.OR_interval[1]:
            break

        sg = copy.deepcopy(subgroup_chosen)
        ant_path.append(actions[action_id])

    q_global = functions.wracc_measure(ID_base, PD_map[ctx.sdt.target[1]], sg.IS, sg.PS)
    q_local = functions.wracc_measure(ctx.sdt.ID, ctx.sdt.PD, sg.IS, sg.PS)
    q_mixed = 0.5 * q_global + 0.5 * q_local
    return sg, ant_path, [q_mixed, q_local, q_global]


# =========================================================
# Runner principal ACS
# =========================================================

def acs_qlearning_pruning_runner(
    initial_seeds_dict: dict,
    seeds: np.ndarray,
    ID_base: int,
    PD_map: dict,
) -> dict:
    start_time = time.time()
    best_global: Pattern | None = None
    best_quality = [0.0, 0.0, 0.0]
    best_path: list = []
    no_improve_count = 0
    no_more_seeds = False
    sqrt_ID = math.sqrt(ctx.sdt.ID)
    actions = ctx.actions.id_to_action

    seeds, initial_fero = evaluate_seeds(seeds, actions, sqrt_ID)
    time_l = [0.0]
    actions_l = [int(seeds.sum())]
    print(f"Initial number of actions={seeds.sum()}")

    ctx.acs.init_params(value=initial_fero)

    for it in range(ctx.sdt.n_iters):
        improve = False
        if no_improve_count >= EARLY_STOP_NO_IMPROVE or no_more_seeds:
            break

        best_sg_iter: Pattern | None = None
        best_path_this_iter: list = []
        best_quality_this_iter = [0.0, 0.0, 0.0]
        
        for cid, info in initial_seeds_dict.items():
            for ant_idx in range(info["n_ants"]):
                if not seeds.any():
                    no_more_seeds = True
                    break
                sg, ant_path, quality_final = _run_ant(
                    info, ant_idx, seeds, actions, best_path, ID_base, PD_map
                )

                q_global = quality_final[2]
                q_local = quality_final[1]

                if q_global >= 0 and q_local > best_quality_this_iter[1]:
                    best_sg_iter = copy.deepcopy(sg)
                    best_path_this_iter = ant_path
                    best_quality_this_iter = quality_final

                    if q_local > best_quality[1]:
                        for idx, active in enumerate(seeds):
                            if not active:
                                continue
                            sg_temp = Pattern(actions[idx])
                            if not functions.evaluate_continue_path(
                                ctx.sdt.ID, ctx.sdt.PD, sg_temp, q_local, ctx.actions
                            ):
                                seeds[idx] = 0

        best_iter_q = best_quality_this_iter[0]
        if best_iter_q > 0 and best_sg_iter is not None:
            if (
                best_iter_q == best_quality[0]
                and not np.array_equal(best_sg_iter.state, best_global.state if best_global else None)
                and np.count_nonzero(best_sg_iter.state) > np.count_nonzero(best_global.state if best_global else np.array([]))
            ):
                best_global = copy.deepcopy(best_sg_iter)
                best_path = best_path_this_iter
            elif best_iter_q > best_quality[0]:
                best_global = copy.deepcopy(best_sg_iter)
                best_quality = best_quality_this_iter
                best_path = best_path_this_iter
                improve = True

        ctx.acs.update_feromonas_global(best_path, best_quality[0])
        no_improve_count = 0 if improve else no_improve_count + 1

        elapsed = time.time() - start_time
        print(
            f"[Iter {it}/{ctx.sdt.n_iters}] best_quality={best_quality[2]:.5f} "
            f"elapsed={elapsed:.1f}s actions={seeds.sum()}"
        )
        time_l.append(elapsed)
        actions_l.append(int(seeds.sum()))

    return {
        "best_solution": best_global,
        "best_quality": best_quality,
        "n_iters": it + 1,
        "time": time_l,
        "actions": actions_l,
    }


# =========================================================
# Setup / Reset
# =========================================================

def _init_context(
    data: pd.DataFrame, categorical_cols: list[str], target: list
) -> None:
    ctx.sdt.init_params(data, categorical_cols, target)
    ctx.states.init_params()
    ctx.actions.init_params()


def setup(data: pd.DataFrame, categorical_cols: list[str], target: list) -> None:
    _init_context(data, categorical_cols, target)


def reset(data: pd.DataFrame, categorical_cols: list[str], target: list) -> None:
    _init_context(data, categorical_cols, target)


# =========================================================
# Loop principal de descubrimiento
# =========================================================

def _best_target_switch(PD_map: dict, ID_base: int) -> str | None:
    """Selecciona el target con mayor exceso de positividad respecto al baseline."""
    best_score = 0.0
    target_best = None
    for key in PD_map:
        if key == ctx.sdt.target[1]:
            continue
        ctx.sdt.target[1] = key
        ctx.sdt.calculate_constants()
        score = (ctx.sdt.PD / ctx.sdt.ID) - (PD_map[key] / ID_base)
        if score > best_score:
            target_best = key
            best_score = score
    return target_best


def main(ID_base: int, PD_map: dict, change_target: bool = True) -> tuple:
    results = []
    start = time.perf_counter()
    mask = np.ones(ctx.actions.n_actions, dtype=int)
    p = (ctx.sdt.PD / ctx.sdt.ID)
    entropy_ini = entropy([p, 1 - p], base=2)
    while ctx.sdt.ID > 10:
        #initial_seeds_dict = fase1.main(ctx.sdt.data, ctx.sdt.target, ctx.actions, cluster=True)
        #initial_seeds = [ctx.actions.translate(col, value) for col, value in (cond.split(":") for cond in initial_seeds_dict["condition"])]
        initial_seeds_dict = fase1.main(ctx.sdt.data, ctx.sdt.target, ctx.actions, cluster=True)
        out = acs_qlearning_pruning_runner(
            initial_seeds_dict=initial_seeds_dict, seeds=mask,
            ID_base=ID_base, PD_map=PD_map
        )

        if (not out["best_solution"]) or (out["best_solution"].IS < math.sqrt(ID_base)):
            break

        results.append((out["best_solution"], ctx.sdt.target[1]))
        instances_covered = functions.rule_to_index_mask(out["best_solution"].state, ctx.sdt)
        ctx.sdt.data = ctx.sdt.data.loc[
            ~ctx.sdt.data.index.isin(ctx.sdt.data[instances_covered].index)
        ]
        ctx.sdt.calculate_constants()
        if change_target:
            pos_rem = ctx.sdt.PD / ctx.sdt.ID
            if pos_rem - (PD_map[ctx.sdt.target[1]] / ID_base) < 0:
                new_target = _best_target_switch(PD_map, ID_base)
                ctx.sdt.target[1] = new_target
        ctx.sdt.calculate_constants()
        mask = ctx.actions.update_action_space()
        p_residuo = ctx.sdt.PD / ctx.sdt.ID
        population_entropy = entropy([p_residuo, 1 - p_residuo], base=2)
        asc = (ctx.sdt.ID / ID_base) * (population_entropy/entropy_ini)
        if population_entropy == 0.0 or asc < MIN_ENTROPY_THRESHOLD:
            break
    elapsed = time.perf_counter() - start
    return elapsed, results


# =========================================================
# Ejecución por dataset
# =========================================================

_STATS_COLUMNS = [
    "n_sub", "length",
    "wracc_l_sum", "wracc_l_mean",
    "coverage_l_sum", "coverage_l_mean",
    "OR_list",
    "wracc_set", "coverage_set", "OR_set",
    "confidence", "PS", "time",
]


def execute(
    df: pd.DataFrame,
    name_df: str,
    target: list,
    n_execs: int = 2,
) -> None:
    stats = []
    setup(df, list(df.columns)[:-1], target)
    PD_map = ctx.sdt.get_map_positives()

    if not target[1]:
        name_file = "dynamic_target"
        target = ["Class", max(PD_map, key=PD_map.get)]
        ctx.sdt.target = target.copy()
        change_target = True
    else:
        name_file = f"ACO_{target[1]}"
        change_target = False

    ctx.sdt.calculate_constants()
    og_task = copy.deepcopy(ctx.sdt)
    ID_base = ctx.sdt.ID

    for exec in range(1, n_execs):
        np.random.seed(exec)
        random.seed(exec)
        time_ej, results = main(ID_base, PD_map, change_target=change_target)
        l_stats = functions.generate_df_custom(results, PD_map, ID_base, og_task)
        l_stats.append(time_ej)
        stats.append(l_stats)
        reset(df, list(df.columns)[:-1], target)
        ctx.sdt.calculate_constants()

    df_stats = pd.DataFrame(data=stats, columns=_STATS_COLUMNS)
    df_stats.loc["Mean"] = df_stats.mean()
    df_stats.to_csv(f"./results/{name_df}_{name_file}.csv")


# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    SINGLE_TARGET = False
    CLASS_COL = "Class"
    N_EXECS = 6
    SPECIAL_DATASETS = {"car", "flare", "nursery"}

    for dataset in os.listdir("../datasets/"):
        #dataset = "splice"
        stats_dir = f"../datasets/{dataset}/stats/"
        os.makedirs(stats_dir, exist_ok=True)

        name_csv = f"{dataset}_for_FSSD" if dataset in SPECIAL_DATASETS else dataset
        df = pd.read_csv(f"../datasets/{dataset}/{name_csv}.csv").astype(str)

        if SINGLE_TARGET:
            for target_value in df[CLASS_COL].unique():
                execute(df, dataset, target=[CLASS_COL, target_value], n_execs=N_EXECS)
        else:
            execute(df, dataset, target=[CLASS_COL, None], n_execs=N_EXECS)
            #exit(0)
