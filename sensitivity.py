"""Local (forward) sensitivity analysis of the network ODEs.

The network is

    dq/dt = phi(w(t, q, p); b, u) - delta * q

where w = NetworkEquations.interaction_terms(...) are the 155 probabilistic
logic terms, phi is the activation function (sigmoid / tanh / hill), b = p[0]
is the saturation rate, u the thresholds and delta the decay rates.

For a set of parameters p_j (any of the 43 network inputs) the sensitivities
S_ij(t) = dq_i/dp_j obey the forward sensitivity equations

    dS/dt = (diag(phi'(w)) J_wq - diag(delta)) S + diag(phi'(w)) J_wp + dphi/db e_b

with S(0) = 0. J_wq = dw/dq and J_wp = dw/dp are obtained symbolically (sympy)
from the same `interaction_terms` the simulator integrates, so the equations
are never duplicated. q and S are integrated together ("augmented system").

See docs/sensitivity_analysis.md for the derivation and how this maps to the
original notebook (sensitivityanalysis.ipynb).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ELASTICITY_MODES = ['relative_pointwise', 'relative_final', 'scaled', 'raw']
DEFAULT_T_SPAN = (0.0, 50.0)
DEFAULT_N_POINTS = 500
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12
HILL_N = 10       # must match create_ode_system.hill_activation
HILL_KD = 0.5

# Inputs set on top of NetworkEquations.initial_out in the thesis notebook
# before each case's own `env_params` are applied.
CASE_BASE_INPUTS = {'LPSE': 1.0, 'IFNGE': 1.0}

# Non-zero initial conditions used by sensitivityanalysis.ipynb, {node index: value}.
# They differ from NetworkEquations.initial_conditions, which sets indices
# 26, 90, 92, 148, 152 (GSK3B, MAMPK, MOXPHOS, AMPK, OXPHOS); the notebook sets
# 26, 88, 89, 150, 154 (GSK3B, MMTORC2, MLKB1, GLUTAMINOLISIS, HIF1A).
# Use `notebook_initial_conditions()` to reproduce the thesis figures exactly.
NOTEBOOK_INITIAL_STATE = {26: 1.0, 88: 1.0, 89: 0.2, 150: 1.0, 154: 0.2}

# The microenvironment cases run in sensitivityanalysis.ipynb (thesis figures
# 5.15-5.20). `range` = (start, stop, step) of the swept parameter.
MICROENVIRONMENT_CASES = [
    {'label': 'Figura_5.15', 'env_params': {'LPSE': 1.0, 'IFNGE': 1.0},
     'node_pair': ('TBET', 'IFNG'), 'sweep_param': 'IFNGE', 'range': (0.9, 1.1, 0.1)},
    {'label': 'Figura_5.16', 'env_params': {'LPSE': 1.0, 'IFNGE': 1.0},
     'node_pair': ('TBET', 'IFNG'), 'sweep_param': 'IFNGE', 'range': (0.98, 1.01, 0.01)},
    {'label': 'Figura_5.17', 'env_params': {'LPSE': 1.0, 'SSRNA': 1.0},
     'node_pair': ('TBET', 'IFNG'), 'sweep_param': 'IFNGE', 'range': (0.9, 1.1, 0.1)},
    {'label': 'Figura_5.18', 'env_params': {'LPSE': 1.0, 'SSRNA': 1.0, 'IL4L': 1.0},
     'node_pair': ('TBET', 'IFNG'), 'sweep_param': 'IFNGE', 'range': (0.65, 0.85, 0.1)},
    {'label': 'Figura_5.19', 'env_params': {'LPSE': 1.0, 'SSRNA': 1.0, 'IL4L': 1.0},
     'node_pair': ('GATA3', 'IL4'), 'sweep_param': 'IFNGE', 'range': (0.65, 0.85, 0.1)},
    {'label': 'Figura_5.20',
     'env_params': {'LPSE': 1.0, 'SSRNA': 1.0, 'IL4L': 1.0, 'IL21E': 1.0, 'TGFBE': 1.0},
     'node_pair': ('RORGT', 'IL21'), 'sweep_param': 'IFNGE', 'range': (0.65, 0.85, 0.1)},
]


# ------------------------------------------------------------
# Activation function and its derivatives
# ------------------------------------------------------------
def activation(af: str, x, b: float, u):
    """Same definitions as NetworkEquations.create_ode_system.act_f."""
    if af in (None, 'sigmoid'):
        return 1.0 / (1.0 + np.exp(-b * (x - u)))
    if af == 'tanh':
        return (1.0 + np.tanh(-b * (x - u))) / 2.0
    if af == 'hill':
        return x ** HILL_N / (HILL_KD ** HILL_N + x ** HILL_N)
    raise ValueError(f"Unknown activation function: {af}")


def activation_derivatives(af: str, x, b: float, u):
    """(dphi/dx, dphi/db) evaluated element-wise at x = w."""
    if af in (None, 'sigmoid'):
        s = activation('sigmoid', x, b, u)
        ds = s * (1.0 - s)
        return b * ds, (x - u) * ds
    if af == 'tanh':
        sech2 = 1.0 - np.tanh(-b * (x - u)) ** 2
        return -0.5 * b * sech2, -0.5 * (x - u) * sech2
    if af == 'hill':
        kn = HILL_KD ** HILL_N
        dx = HILL_N * kn * x ** (HILL_N - 1) / (kn + x ** HILL_N) ** 2
        return dx, np.zeros_like(x)          # hill does not depend on b
    raise ValueError(f"Unknown activation function: {af}")


def notebook_initial_conditions(n: int = 155) -> np.ndarray:
    q0 = np.zeros(n)
    for idx, value in NOTEBOOK_INITIAL_STATE.items():
        q0[idx] = value
    return q0


def sweep_values(start: float, stop: float, step: float) -> np.ndarray:
    """start, start+step, ..., stop (inclusive, as in the notebook)."""
    if step <= 0:
        raise ValueError("The sweep step must be positive.")
    if stop < start:
        raise ValueError("The sweep end must be >= the sweep start.")
    return np.round(np.arange(start, stop + 0.5 * step, step), 10)


@dataclass
class SensitivityResult:
    """Trajectories and sensitivities of one sweep.

    q -> (n_sweep, n_nodes, n_t)            node activities
    S -> (n_sweep, n_nodes, n_params, n_t)  dq_i/dp_j
    p -> (n_sweep, n_inputs)                inputs used in each run
    """
    t: np.ndarray
    q: np.ndarray
    S: np.ndarray
    p: np.ndarray
    param_names: List[str]
    param_indices: List[int]
    sweep_param: Optional[str]
    sweep_values: np.ndarray
    node_names: List[str]

    def node_index(self, node: str) -> int:
        return self.node_names.index(node)

    def param_position(self, param: str) -> int:
        return self.param_names.index(param)


class SensitivityAnalyzer:
    """Forward sensitivity analysis on top of a NetworkEquations instance."""

    def __init__(self, network):
        self.network = network
        self.n = network.n
        self.m = len(network.out_of_network_arguments)
        self.node_names = [str(x) for x in network.array_of_elements]
        self.input_names = [str(x) for x in network.out_of_network_arguments]
        self._jac_rows = None
        self._jac_cols = None
        self._jac_fn = None

    # ------------------------------------------------------------
    # Symbolic Jacobian of w with respect to [q | p] (sparse)
    # ------------------------------------------------------------
    def build_jacobian(self) -> None:
        """Differentiate the 155 terms once and lambdify the non-zero entries.

        Only ~1 % of the 155 x 198 entries are structurally non-zero, so the
        Jacobian is stored as (rows, cols, values) and scattered into a dense
        array at every RHS call.
        """
        if self._jac_fn is not None:
            return
        t = sp.Symbol('t')
        q = sp.symbols(f'q0:{self.n}')
        p = sp.symbols(f'p0:{self.m}')
        w = self.network.interaction_terms(t, q, p, [0] * self.n, exp=sp.exp)
        position = {v: j for j, v in enumerate(list(q) + list(p))}

        rows, cols, exprs = [], [], []
        for i, wi in enumerate(w):
            wi = sp.sympify(wi)
            for v in sorted(wi.free_symbols - {t}, key=position.get):
                rows.append(i)
                cols.append(position[v])
                exprs.append(sp.diff(wi, v))

        self._jac_rows = np.asarray(rows)
        self._jac_cols = np.asarray(cols)
        self._jac_fn = sp.lambdify([t, *q, *p], exprs, 'numpy', cse=True)

    def jacobian(self, t, q, p) -> np.ndarray:
        """Dense dw/d[q | p] of shape (n, n + m)."""
        self.build_jacobian()
        J = np.zeros((self.n, self.n + self.m))
        J[self._jac_rows, self._jac_cols] = self._jac_fn(t, *q, *p)
        return J

    # ------------------------------------------------------------
    # Right-hand sides
    # ------------------------------------------------------------
    def _augmented_rhs(self, af, decay, thresholds, param_indices):
        """RHS of y = [q | vec(S)], with S of shape (n, k)."""
        n, k = self.n, len(param_indices)
        ode = self.network.create_ode_system(af, decay, thresholds)
        p_cols = n + np.asarray(param_indices)
        b_column = [c for c, idx in enumerate(param_indices) if idx == 0]
        diag = np.diag_indices(n)

        def rhs(t, y, *p):
            q = y[:n]
            S = y[n:].reshape(n, k)
            w = self.network.interaction_terms(t, q, p, np.zeros(n))
            J = self.jacobian(t, q, p)
            dphi_dw, dphi_db = activation_derivatives(af, w, p[0], thresholds)
            A = dphi_dw[:, None] * J[:, :n]
            A[diag] -= decay
            B = dphi_dw[:, None] * J[:, p_cols]
            for c in b_column:                 # b also enters phi directly
                B[:, c] += dphi_db
            dS = A @ S + B
            return np.concatenate([ode(t, q, *p), dS.ravel()])

        return rhs

    # ------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------
    def _defaults(self, q0, decay, thresholds):
        q0 = self.network.initial_conditions if q0 is None else q0
        decay = np.ones(self.n) if decay is None else decay
        thresholds = np.full(self.n, 0.5) if thresholds is None else thresholds
        return (np.asarray(q0, dtype=float), np.asarray(decay, dtype=float),
                np.asarray(thresholds, dtype=float))

    def solve_base(self, p, q0=None, decay=None, thresholds=None, af='sigmoid',
                   t_span=DEFAULT_T_SPAN, n_points=DEFAULT_N_POINTS,
                   rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, method='LSODA'):
        """Plain network ODE (no sensitivities), with tight tolerances."""
        q0, decay, thresholds = self._defaults(q0, decay, thresholds)
        ode = self.network.create_ode_system(af, decay, thresholds)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(ode, t_span, q0, method=method, t_eval=t_eval,
                        args=tuple(np.asarray(p, dtype=float)),
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        return sol.t, sol.y

    def solve_augmented(self, p, param_names, q0=None, decay=None, thresholds=None,
                        af='sigmoid', t_span=DEFAULT_T_SPAN,
                        n_points=DEFAULT_N_POINTS, rtol=DEFAULT_RTOL,
                        atol=DEFAULT_ATOL, method='LSODA'):
        """One augmented integration. Returns t, q (n, n_t), S (n, k, n_t)."""
        q0, decay, thresholds = self._defaults(q0, decay, thresholds)
        param_indices = [self.input_names.index(name) for name in param_names]
        k = len(param_indices)
        rhs = self._augmented_rhs(af, decay, thresholds, param_indices)
        y0 = np.concatenate([q0, np.zeros(self.n * k)])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval,
                        args=tuple(np.asarray(p, dtype=float)),
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        q = sol.y[:self.n]
        S = sol.y[self.n:].reshape(self.n, k, -1)
        return sol.t, q, S

    def run(self, p_nominal, param_names: Sequence[str],
            sweep_param: Optional[str] = None, values: Optional[Sequence[float]] = None,
            progress=None, **solver_kwargs) -> SensitivityResult:
        """Sensitivities w.r.t. `param_names` at every value of `sweep_param`.

        Without a sweep the analysis runs once at `p_nominal` (the inputs
        chosen in the interface).
        """
        if not param_names:
            raise ValueError("Select at least one parameter.")
        p_nominal = np.asarray(p_nominal, dtype=float)
        if sweep_param is None:
            values = np.array([np.nan])
        else:
            values = np.asarray(values, dtype=float)
        sweep_idx = None if sweep_param is None else self.input_names.index(sweep_param)

        q_all, S_all, p_all = [], [], []
        for i, v in enumerate(values):
            p = p_nominal.copy()
            if sweep_idx is not None:
                p[sweep_idx] = v
            t, q, S = self.solve_augmented(p, param_names, **solver_kwargs)
            q_all.append(q)
            S_all.append(S)
            p_all.append(p)
            if progress is not None:
                progress((i + 1) / len(values))

        return SensitivityResult(
            t=t, q=np.stack(q_all), S=np.stack(S_all), p=np.stack(p_all),
            param_names=list(param_names),
            param_indices=[self.input_names.index(x) for x in param_names],
            sweep_param=sweep_param, sweep_values=values,
            node_names=self.node_names,
        )

    # ------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------
    @staticmethod
    def elasticity(result: SensitivityResult, node: str, param: str,
                   mode: str = 'relative_pointwise', eps: float = 1e-12) -> np.ndarray:
        """Normalised sensitivity of `node` to `param`, shape (n_sweep, n_t).

        relative_pointwise : (dq/dp) * p / |q(t)|
        relative_final     : (dq/dp) * p / |q(T)|
        scaled             : (dq/dp) * p
        raw                : dq/dp
        """
        i = result.node_index(node)
        j = result.param_position(param)
        s = result.S[:, i, j, :]
        q = result.q[:, i, :]
        p_val = result.p[:, result.param_indices[j]][:, None]
        if mode == 'relative_pointwise':
            return s * p_val / np.maximum(np.abs(q), eps)
        if mode == 'relative_final':
            return s * p_val / np.maximum(np.abs(q[:, -1:]), eps)
        if mode == 'scaled':
            return s * p_val
        if mode == 'raw':
            return s
        raise ValueError(f"Unknown elasticity mode: {mode}")

    def summary(self, result: SensitivityResult, nodes: Sequence[str],
                mode: str = 'relative_pointwise') -> List[Dict]:
        """Min/max of activity, sensitivity and elasticity per (node, parameter)."""
        rows = []
        for node in nodes:
            i = result.node_index(node)
            q = result.q[:, i, :]
            for param in result.param_names:
                j = result.param_position(param)
                s = result.S[:, i, j, :]
                e = self.elasticity(result, node, param, mode)
                rows.append({
                    'node': node, 'parameter': param,
                    'activity_min': float(q.min()), 'activity_max': float(q.max()),
                    'sensitivity_min': float(s.min()), 'sensitivity_max': float(s.max()),
                    'elasticity_min': float(e.min()), 'elasticity_max': float(e.max()),
                    'elasticity_final': float(e[-1, -1]),
                })
        return rows

    def finite_difference_check(self, p, param: str, node: str,
                                deltas: Sequence[float] = (1e-2, 1e-3, 1e-4),
                                q0=None, decay=None, thresholds=None,
                                af='sigmoid', t_span=DEFAULT_T_SPAN,
                                n_points=DEFAULT_N_POINTS, rtol=DEFAULT_RTOL,
                                atol=DEFAULT_ATOL, method='LSODA') -> Dict:
        """Compare dq/dp from the sensitivity ODE with central differences.

        Also reports max|q_augmented - q_base|, which checks that adding the
        sensitivity states does not change the trajectory of q.
        """
        kw = dict(q0=q0, decay=decay, thresholds=thresholds, af=af, t_span=t_span,
                  n_points=n_points, rtol=rtol, atol=atol, method=method)
        p = np.asarray(p, dtype=float)
        j = self.input_names.index(param)
        i = self.node_names.index(node)

        t, q_aug, S = self.solve_augmented(p, [param], **kw)
        _, q_base = self.solve_base(p, **kw)

        fd = {}
        for d in deltas:
            p_plus, p_minus = p.copy(), p.copy()
            p_plus[j] += d
            p_minus[j] -= d
            _, y_plus = self.solve_base(p_plus, **kw)
            _, y_minus = self.solve_base(p_minus, **kw)
            fd[d] = (y_plus[i] - y_minus[i]) / (2 * d)

        return {'t': t, 'exact': S[i, 0], 'fd': fd, 'node': node, 'param': param,
                'augmented_vs_base': float(np.max(np.abs(q_aug[i] - q_base[i])))}

    def case_inputs(self, env_params: Dict[str, float],
                    base_inputs: Dict[str, float] = None) -> np.ndarray:
        """Input vector of a notebook case: defaults + base inputs + env_params."""
        p = np.asarray(self.network.initial_out, dtype=float).copy()
        for name, value in {**(base_inputs if base_inputs is not None else CASE_BASE_INPUTS),
                            **env_params}.items():
            p[self.input_names.index(name)] = value
        return p
