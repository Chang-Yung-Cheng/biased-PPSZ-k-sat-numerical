"""Minimal readable runner for the generic k-SAT numerical experiment.

The core numerical formulas are preserved.  Finalization now evaluates only the final active t-set with NO-R_k solves.

Only tests, verbose progress reporting, JSON output, and unused/dead default-path machinery are removed.

k=7 is added as an extension.

"""

from __future__ import annotations

import math

from decimal import Decimal, ROUND_CEILING

from dataclasses import dataclass

from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp

import numpy as np

from scipy.integrate import quad

from scipy.optimize import brentq  # 找方程根

LN2 = math.log(2.0)

SUPPORTED_K = (3, 4, 5, 6, 7)

QUAD_EPSABS = 2e-13

QUAD_EPSREL = 2e-13

ROOT_XTOL = 5e-15

CLARABEL_MAX_ITER = 2500

SCS_MAX_ITERS = 120000

T_STEP = 0.005  # scan t step

ACTIVE_STEP = 0.05  # initial t step

BISECTION_ITERS = 50  # r bisection

DUPLICATE_TOL = 0.0002  # epsilon t

GOLDEN_TOL = 2e-07  # epsilon g

GOLDEN_ITERS = 80  # Golden Search

MAX_LOCAL_REFINEMENTS = 3

# ==============================================================================

# 1. k-dependent PPSZ numerics

# ==============================================================================

@dataclass

class KNumerics:

    k: int

    t_critical: float

    S_k: float

    j_turn: float

    def Solve_d(self, p: float) -> float:

        """Smallest solution d of d = p + (1-p)d^(k-1)."""

        k = self.k

        if not 0.0 <= p <= 1.0:

            raise ValueError('p must lie in [0,1]')

        if p <= 0.0:

            return 0.0

        if p >= self.t_critical:

            return 1.0

        d_turn = (1.0 / ((k - 1) * (1.0 - p))) ** (1.0 / (k - 2))  # f'(d)=0 的轉折點

        def f(d: float) -> float:

            return p + (1.0 - p) * d ** (k - 1) - d  # f(d)=0

        return brentq(f, 0.0, d_turn, xtol=ROOT_XTOL, rtol=4.0 * np.finfo(float).eps, maxiter=200)  # 找最小根

    def Q_k(self, p: float) -> float:

        d = self.Solve_d(p)  # d_k(p)

        return d ** (self.k - 1)  # q_k(p)

    def R_k(self, t: float) -> float:

        """R_k(t)=int_0^t q_k(p) dp, numerically."""

        if not 0.0 <= t <= 1.0:

            raise ValueError('t must lie in [0,1]')

        if t <= 0.0:

            return 0.0

        if t <= self.t_critical:

            val, _ = quad(lambda p: self.Q_k(p), 0.0, t, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=300)  # 積分 R_k(t)

            return float(val)

        rc, _ = quad(lambda p: self.Q_k(p), 0.0, self.t_critical, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=300)  # 積到臨界點

        return float(rc + (t - self.t_critical))

    def J_integrand(self, p: float) -> float:

        return 2.0 * p ** (self.k - 1) - p ** self.k - self.Q_k(p)  # J_k 的積分

    def J_k(self, t: float) -> float:

        """J_k(t)=int_0^t max(2p^(k-1)-p^k-q_k(p),0) dp."""

        if not 0.0 <= t <= 1.0:

            raise ValueError('t must lie in [0,1]')

        u = min(t, self.j_turn)  # J_k 積正區間

        if u <= 0.0:

            return 0.0

        val, _ = quad(lambda p: max(self.J_integrand(p), 0.0), 0.0, u, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=300)  # 積分 J_k(t)

        return float(val)

def Build_K_Numerics(k: int) -> KNumerics:

    if k not in SUPPORTED_K:

        raise ValueError(f'supported k values are {SUPPORTED_K}')

    tcrit = (k - 2.0) / (k - 1.0)  # t 臨界點

    tmp = KNumerics(k=k, t_critical=tcrit, S_k=float('nan'), j_turn=float('nan'))  # 暫存 k 數值

    rcrit, _ = quad(lambda p: tmp.Q_k(p), 0.0, tcrit, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=400)  # R_k(t_c)

    S_k = float(tcrit - rcrit)  # S_k

    xs = np.linspace(1e-08, tcrit * (1.0 - 1e-10), 3000)  # 粗掃 j_turn

    vals = [tmp.J_integrand(float(x)) for x in xs]  # 各點 surplus

    bracket = None

    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):

        if fa > 0.0 and fb <= 0.0:

            bracket = (float(a), float(b))  # 找正轉負區間

            break

    if bracket is None:

        raise RuntimeError(f'could not bracket positive J_k turning point for k={k}')

    j_turn = brentq(lambda p: tmp.J_integrand(p), bracket[0], bracket[1], xtol=ROOT_XTOL, rtol=4.0 * np.finfo(float).eps, maxiter=200)  # 精確找 j_turn

    tmp.j_turn = float(j_turn)

    return KNumerics(k=k, t_critical=float(tcrit), S_k=S_k, j_turn=float(j_turn))

# ==============================================================================

# 2. Prefix system and strengthened caps

# ==============================================================================

def Prefix_Set(k: int) -> Tuple[str, ...]:

    """

    Qin-style prefixes: empty, all binary strings of length 1..k-2, and

    all length-(k-1) strings except 0^(k-1).  Thus |P|=2^k-2.

    """

    out: List[str] = ['empty']

    for ell in range(1, k):  # prefix 長度 1 到 k-1

        for x in range(2 ** ell):

            s = format(x, f'0{ell}b')

            if ell == k - 1 and s == '0' * (k - 1):  # 排除 0^(k-1)

                continue

            out.append(s)

    return tuple(out)

def Prefix_Length(p: str) -> int:

    return 0 if p == 'empty' else len(p)

def Prefix_Integral(k: int, t: float) -> np.ndarray:

    """

    I_r(t)=int_0^t C(k-1,r)p^r(1-p)^(k-1-r) dp, r=0..k-1.

    Polynomial evaluation is exact up to floating-point roundoff and avoids

    numerical quadrature for these simple small-k integrals.

    """

    vals = np.zeros(k, dtype=float)  # 存 I_r(t)

    n = k - 1  # 二項式次數

    for r in range(k):

        total = 0.0

        for j in range(n - r + 1):

            total += math.comb(n, r) * math.comb(n - r, j) * (-1.0) ** j * t ** (r + j + 1) / (r + j + 1)  # 直接積分

        vals[r] = total

    return vals

def Weight_Coefficient(k: int, clause_weight: int, bits: str) -> float:

    """Probability coefficient of a specific ordered partial 0/1 pattern."""

    m = len(bits)  # 固定位置數

    ones = bits.count('1')  # 已出現的 1

    zeros = m - ones  # 已出現的 0

    if ones > clause_weight or zeros > k - clause_weight:

        return 0.0

    remain_ones = clause_weight - ones  # 剩下要放的 1

    remain_slots = k - m  # 剩餘位置

    if remain_ones < 0 or remain_ones > remain_slots:

        return 0.0

    return math.comb(remain_slots, remain_ones) / math.comb(k, clause_weight)  # pattern 係數

@dataclass

class PrefixSystem:

    k: int

    prefixes: Tuple[str, ...]

    lengths: np.ndarray

    M0: np.ndarray

    M1: np.ndarray

    full_prefix_indices: Tuple[int, ...]

    @property

    def n_types(self) -> int:

        return len(self.prefixes)

def Build_Prefix_System(k: int) -> PrefixSystem:

    prefixes = Prefix_Set(k)  # 建所有 prefix

    lengths = np.array([Prefix_Length(p) for p in prefixes], dtype=int)  # prefix 長度

    n_types = len(prefixes)

    M0 = np.zeros((n_types, k), dtype=float)  # b=0 係數矩陣

    M1 = np.zeros((n_types, k), dtype=float)  # b=1 係數矩陣

    for j, p in enumerate(prefixes):

        bits = '' if p == 'empty' else p

        for i in range(1, k + 1):

            M0[j, i - 1] = Weight_Coefficient(k, i, bits + '0')

            M1[j, i - 1] = Weight_Coefficient(k, i, bits + '1')

    full = tuple((j for j, p in enumerate(prefixes) if Prefix_Length(p) == k - 1))  # 長度 k-1 的 prefix

    return PrefixSystem(k, prefixes, lengths, M0, M1, full)

def Build_v_Matrices(ps: PrefixSystem, t: float) -> Tuple[np.ndarray, np.ndarray, float]:

    ints = Prefix_Integral(ps.k, t)  # 各長度 I_r(t)

    scales = np.array([ints[ell] for ell in ps.lengths], dtype=float)  # 每個 P 對應 I_r(t)

    A0 = scales[:, None] * ps.M0  # v_P^0 係數

    A1 = scales[:, None] * ps.M1  # v_P^1 係數

    special_factor_w1 = ints[ps.k - 1] / ps.k  # w1 forced

    return (A0, A1, float(special_factor_w1))

def Strengthened_Caps_Expr(ps: PrefixSystem, w: cp.Expression, z: cp.Expression, J_k: float) -> Tuple[cp.Expression, cp.Expression]:

    """Affine loss vectors subtracted from v0 and v1."""

    k = ps.k

    zero = cp.Constant(0.0)

    loss0: List[cp.Expression] = [zero for _ in range(ps.n_types)]

    loss1: List[cp.Expression] = [zero for _ in range(ps.n_types)]

    for j in ps.full_prefix_indices:

        p = ps.prefixes[j]

        s = p.count('1')

        i0 = s

        if i0 == 1:

            loss0[j] = w[0] / k * J_k

        elif 2 <= i0 <= k - 1:

            zi = z[i0 - 2]

            denom = (k - i0) * math.comb(k, i0)

            loss0[j] = ((k - i0) * w[i0 - 1] - zi) / denom * J_k

        i1 = s + 1

        if 2 <= i1 <= k - 1:

            zi = z[i1 - 2]

            denom = i1 * math.comb(k, i1)

            loss1[j] = ((i1 - 1) * w[i1 - 1] + zi) / denom * J_k

        elif i1 == k:

            loss1[j] = (k - 1.0) / k * w[k - 1] * J_k

    return (cp.hstack(loss0), cp.hstack(loss1))

def Strengthened_Caps(ps: PrefixSystem, w: np.ndarray, z: np.ndarray, J_k: float) -> Tuple[np.ndarray, np.ndarray]:  # L0, L1

    k = ps.k

    l0 = np.zeros(ps.n_types, dtype=float)

    l1 = np.zeros(ps.n_types, dtype=float)

    for j in ps.full_prefix_indices:  # 只有 full prefixes 有 L

        p = ps.prefixes[j]

        s = p.count('1')

        i0 = s

        if i0 == 1:

            l0[j] = w[0] / k * J_k

        elif 2 <= i0 <= k - 1:

            zi = z[i0 - 2]

            l0[j] = ((k - i0) * w[i0 - 1] - zi) / ((k - i0) * math.comb(k, i0)) * J_k

        i1 = s + 1

        if 2 <= i1 <= k - 1:

            zi = z[i1 - 2]

            l1[j] = ((i1 - 1) * w[i1 - 1] + zi) / (i1 * math.comb(k, i1)) * J_k

        elif i1 == k:

            l1[j] = (k - 1.0) / k * w[k - 1] * J_k

    return (l0, l1)

# ==============================================================================

# 3. Entropy and recursive exponent

# ==============================================================================

def h_expr(g0: cp.Expression, g1: cp.Expression) -> cp.Expression:

    s = g0 + g1

    return (-cp.rel_entr(g0, s) - cp.rel_entr(g1, s)) / LN2  # x*ln(x/y)

def h_numeric(a: float, b: float) -> float:

    a = max(float(a), 0.0)

    b = max(float(b), 0.0)

    s = a + b

    if s <= 1e-18:

        return 0.0

    ans = 0.0

    if a > 0.0:

        ans -= a * math.log2(a / s)

    if b > 0.0:

        ans -= b * math.log2(b / s)

    return ans

def s_w_expr(k: int, w: cp.Expression) -> cp.Expression:  # s_w w unknown

    terms = [cp.rel_entr(w[i - 1], float(math.comb(k, i))) for i in range(1, k + 1)]

    return -cp.sum(cp.hstack(terms)) / LN2

def s_w_numeric(k: int, w: np.ndarray) -> float: # s_w w known

    ans = 0.0

    for i in range(1, k + 1):

        wi = float(w[i - 1])

        if wi > 0.0:

            ans += wi * math.log2(math.comb(k, i) / wi)

    return ans

def E_Balanced(k: int, S_k: float, R_prev: float, sw: float, delta: float) -> dict:  # 直接找交點

    denom = sw + k * delta - k * R_prev

    if denom <= 0.0:

        return {'gamma': float('inf'), 'large_D_exponent': float('nan'), 'small_D_exponent': float('nan'), 'exponent': float('nan'), 'gamma_valid': False}

    gamma = (S_k - R_prev) / denom  # 平衡兩分支

    large = S_k - k * gamma * delta  # large-D exponent

    small = gamma * sw + R_prev * (1.0 - k * gamma)  # small-D exponent

    return {'gamma': float(gamma), 'large_D_exponent': float(large), 'small_D_exponent': float(small), 'exponent': float(max(large, small)), 'gamma_valid': bool(-1e-11 <= gamma <= 1.0 / k + 1e-11)}

def E_Bounded(k: int, S_k: float, R_prev: float, sw: float, delta: float) -> dict:  # 在 [0,1/k] 找最小值

    def A(g: float) -> float:

        return S_k - k * g * delta

    def B(g: float) -> float:

        return g * sw + R_prev * (1.0 - k * g)

    candidates = [0.0, 1.0 / k]

    denom = sw + k * delta - k * R_prev

    gb = float('inf')

    if abs(denom) > 1e-18:

        gb = (S_k - R_prev) / denom

        if 0.0 <= gb <= 1.0 / k:

            candidates.append(gb)

    vals = [(g, max(A(g), B(g))) for g in candidates]

    gbest, ebest = min(vals, key=lambda x: x[1])

    return {'gamma': float(gbest), 'exponent': float(ebest), 'large_D_exponent': float(A(gbest)), 'small_D_exponent': float(B(gbest)), 'balance_gamma': float(gb)}

# ==============================================================================

# 4. t-grid and cached t-data

# ==============================================================================

def Build_Threshold_Set(tcrit: float, step: float) -> List[float]:

    if not 0.0 < step <= tcrit:

        raise ValueError('t step must be in (0,t_critical]')

    vals: List[float] = []  # 建 t grid

    m = int(math.floor((tcrit - 1e-14) / step))

    for j in range(1, m + 1):

        x = j * step

        if x < tcrit - 1e-12:

            vals.append(float(round(x, 14)))

    vals.append(float(tcrit))  # 加入t_c

    out: List[float] = []

    for x in vals:

        if not out or abs(x - out[-1]) > 1e-13:

            out.append(x)

    return out

@dataclass

class TData:

    t: float

    R_k: float

    J_k: float

    A0: np.ndarray

    A1: np.ndarray

    special_factor_w1: float

def Get_Constraint_Data(num: KNumerics, ps: PrefixSystem, cache: Dict[float, TData], t: float) -> TData:

    t = min(num.t_critical, max(0.0, float(t)))

    key = round(t, 14)  # cache key

    if key not in cache:

        A0, A1, special_factor = Build_v_Matrices(ps, t)  # 建 t 的 v 矩陣

        cache[key] = TData(t=t, R_k=num.R_k(t), J_k=num.J_k(t), A0=A0, A1=A1, special_factor_w1=special_factor)

    return cache[key]

# ==============================================================================

# 5. WITH-R_k joint CVXPY model and numerical feasibility classification

# ==============================================================================

@dataclass

class WithRJointModel:

    problem: cp.Problem

    w: cp.Variable

    z: cp.Variable  # z_i = rho_i * w_i

    a: cp.Parameter  # S_k - r

    b: cp.Parameter  # k(r - R_k-1)

    eta: cp.Variable

    blocks: List[dict]  # t data

def Build_With_r_Constraints(k: int, R_prev: float, ps: PrefixSystem, tdata: Sequence[TData]) -> WithRJointModel:  # 建立 model

    """WITH-R_k: trial r is fixed; solve the shared w,rho problem on the active t-set."""

    w = cp.Variable(k, nonneg=True, name='w')  # 長度 k, >=0

    z = cp.Variable(k - 2, nonneg=True, name='z_rho_times_w')  # 長度 k-2, >=0

    a = cp.Parameter(nonneg=True, name='a_S_minus_r')

    b = cp.Parameter(nonneg=True, name='b_k_r_minus_Rprev')

    eta = cp.Variable(name='eta')

    constraints: List[cp.Constraint] = [cp.sum(w) == 1.0]  # w sum = 1

    for i in range(2, k):

        constraints.append(z[i - 2] <= w[i - 1])  # zi <= wi

    sw = s_w_expr(k, w)

    blocks: List[dict] = []

    for idx, td in enumerate(tdata, 1):

        v0 = td.A0 @ w  # 係數 * w

        v1 = td.A1 @ w

        special = td.special_factor_w1 * w[0]  # P all 0, b = 1

        f0 = cp.Variable(ps.n_types, nonneg=True, name=f'f0_t{idx:03d}')  # 長度 |{P}|, >=0

        f1 = cp.Variable(ps.n_types, nonneg=True, name=f'f1_t{idx:03d}')

        loss0, loss1 = Strengthened_Caps_Expr(ps, w, z, td.J_k)  # strengthened cap

        hat0 = v0 - loss0  # f0 上界

        hat1 = v1 - loss1  # f1 上界

        constraints += [f0 <= hat0, f1 <= hat1]

        g0 = v0 - f0

        g1 = v1 - f1

        constraints += [g0 >= 0.0, g1 >= 0.0]

        constraints += [special + cp.sum(f0) + cp.sum(f1) == td.R_k]  # f sum = R_k(t)

        H = cp.sum(h_expr(g0, g1))  # guessed entropy

        constraints += [a * (sw - k * R_prev) + b * H - b * (td.t - td.R_k) >= eta]  # r constraints

        blocks.append({'td': td, 'f0': f0, 'f1': f1})

    problem = cp.Problem(cp.Maximize(eta), constraints)  # 最大化r

    if not problem.is_dcp():

        raise RuntimeError('internal error: generic joint problem is not DCP')

    return WithRJointModel(problem, w, z, a, b, eta, blocks)

def Solve_CLARABEL(problem: cp.Problem) -> str:

    try:

        problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, max_iter=CLARABEL_MAX_ITER, tol_gap_abs=1e-09, tol_gap_rel=1e-09, tol_feas=1e-09)  # verbose : 不打印, con, obj, tolerance 誤差

        return str(problem.status)  # 印結果

    except cp.error.SolverError:

        try:

            problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)  #容忍誤差

            return str(problem.status)

        except Exception:

            return 'solver_error'

def Solve_SCS(problem: cp.Problem) -> str:

    if 'SCS' not in cp.installed_solvers():

        return 'scs_not_installed'

    try:

        problem.solve(solver=cp.SCS, warm_start=True, verbose=False, eps_abs=2e-07, eps_rel=2e-07, max_iters=SCS_MAX_ITERS, acceleration_lookback=10, normalize=True)

        return str(problem.status)

    except (cp.error.SolverError, ValueError):

        try:

            problem.solve(solver=cp.SCS, warm_start=True, verbose=False, eps=2e-07, max_iters=SCS_MAX_ITERS)

            return str(problem.status)

        except Exception:

            return 'solver_error'

def Check_With_r_Feasibility(k: int, R_prev: float, ps: PrefixSystem, model: WithRJointModel, a_val: float, b_val: float) -> dict:  # 驗算

    if model.w.value is None or model.eta.value is None:

        return {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': None}

    wv = np.asarray(model.w.value, dtype=float).reshape(k)

    if model.z.value is None:

        return {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': None}

    zv = np.asarray(model.z.value, dtype=float).reshape(k - 2)  # w, z 轉 np

    eta_val = float(model.eta.value)

    if not np.all(np.isfinite(wv)) or not np.all(np.isfinite(zv)) or (not math.isfinite(eta_val)):

        return {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': eta_val}

    worst = max(float(np.max(np.maximum(-wv, 0.0))), abs(float(wv.sum()) - 1.0))  # 驗算 w >= 0, w sum = 1

    worst = max(worst, float(np.max(np.maximum(-zv, 0.0))))  # 驗算 z >= 0

    if len(zv):

        worst = max(worst, float(np.max(np.maximum(zv - wv[1:k - 1], 0.0))))  # 驗算 z <= w

    sw_num = s_w_numeric(k, wv)

    margins: List[float] = []

    for block in model.blocks:

        td: TData = block['td']

        if block['f0'].value is None or block['f1'].value is None:

            return {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': eta_val}

        f0 = np.asarray(block['f0'].value, dtype=float).reshape(ps.n_types)  # f 轉 np

        f1 = np.asarray(block['f1'].value, dtype=float).reshape(ps.n_types)

        if not np.all(np.isfinite(f0)) or not np.all(np.isfinite(f1)):

            return {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': eta_val}

        v0 = td.A0 @ wv

        v1 = td.A1 @ wv

        special = td.special_factor_w1 * wv[0]

        l0, l1 = Strengthened_Caps(ps, wv, zv, td.J_k)

        hat0, hat1 = (v0 - l0, v1 - l1)

        g0, g1 = (v0 - f0, v1 - f1)

        worst = max(worst, float(np.max(np.maximum(-f0, 0.0))))

        worst = max(worst, float(np.max(np.maximum(-f1, 0.0))))

        worst = max(worst, float(np.max(np.maximum(f0 - hat0, 0.0))))  # 驗算 f <= v^hat

        worst = max(worst, float(np.max(np.maximum(f1 - hat1, 0.0))))

        worst = max(worst, float(np.max(np.maximum(-g0, 0.0))))

        worst = max(worst, float(np.max(np.maximum(-g1, 0.0))))

        forced = float(special + f0.sum() + f1.sum())

        worst = max(worst, abs(forced - td.R_k))

        Hnum = sum((h_numeric(max(g0[j], 0.0), max(g1[j], 0.0)) for j in range(ps.n_types)))

        margin = a_val * (sw_num - k * R_prev) + b_val * Hnum - b_val * (td.t - td.R_k)

        margins.append(float(margin))

    min_margin = min(margins) if margins else float('-inf')

    eta_gap = max(0.0, eta_val - min_margin)

    worst = max(worst, eta_gap)

    return {'valid': math.isfinite(worst) and math.isfinite(min_margin), 'base_violation': float(worst), 'min_margin': float(min_margin), 'eta': eta_val, 'eta_gap': float(eta_val - min_margin)}  # 回傳 dict

def Solve_With_r_Feasibility(k: int, R_prev: float, ps: PrefixSystem, model: WithRJointModel, a_val: float, b_val: float) -> dict:

    attempts = []

    primal_statuses = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT)

    status = Solve_CLARABEL(model.problem)  # solve

    check = Check_With_r_Feasibility(k, R_prev, ps, model, a_val, b_val) if status in primal_statuses else {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': None, 'eta_gap': float('inf')}

    ca = {'solver': 'CLARABEL', 'status': status, **check}

    attempts.append(ca)

    if check['valid']:

        guard = max(5e-08, 20.0 * check['base_violation'])

        if check['min_margin'] >= guard:  # feasible

            return {'classification': 'feasible', **ca, 'attempts': attempts}

        if status == cp.OPTIMAL and check['base_violation'] <= 1e-08 and (check['eta'] is not None):

            if check['eta'] <= -guard:  # infeasible

                return {'classification': 'infeasible', **ca, 'attempts': attempts}

    if status == cp.INFEASIBLE:

        return {'classification': 'infeasible', **ca, 'attempts': attempts}

    status2 = Solve_SCS(model.problem)

    check2 = Check_With_r_Feasibility(k, R_prev, ps, model, a_val, b_val) if status2 in primal_statuses else {'valid': False, 'base_violation': float('inf'), 'min_margin': float('-inf'), 'eta': None, 'eta_gap': float('inf')}

    sa = {'solver': 'SCS', 'status': status2, **check2}

    attempts.append(sa)

    if check2['valid']:

        guard = max(2e-06, 20.0 * check2['base_violation'])

        if check2['min_margin'] >= guard:

            return {'classification': 'feasible', **sa, 'attempts': attempts}

        if status2 == cp.OPTIMAL and check2['base_violation'] <= 2e-07 and (check2['eta'] is not None):

            if check2['eta'] <= -guard:

                return {'classification': 'infeasible', **sa, 'attempts': attempts}

    if status2 == cp.INFEASIBLE:

        return {'classification': 'infeasible', **sa, 'attempts': attempts}

    return {'classification': 'unknown', **sa, 'attempts': attempts}

def Solve_No_r_CVXPY(problem: cp.Problem) -> str:

    st = Solve_CLARABEL(problem)

    if st in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):

        return st

    return Solve_SCS(problem)

def Clean_w_rho(wv: np.ndarray, zv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # 清理誤差

    w = np.array(wv, dtype=float) 

    z = np.array(zv, dtype=float)

    w[np.abs(w) < 1e-12] = 0.0

    z[np.abs(z) < 1e-12] = 0.0

    s = float(w.sum())

    if s > 0.0:

        w /= s

        z /= s

    for j in range(len(z)):

        z[j] = min(max(z[j], 0.0), w[j + 1])

    return (w, z)

# ==============================================================================

# 6. WITH-R_k active t-set -> shared w*, rho*

# ==============================================================================

# WITH-R_k: this is the only stage that uses trial r / bisection.

def With_r_CVXPY(

    k: int,

    num: KNumerics,

    R_prev: float,

    ps: PrefixSystem,

    active_t: Sequence[float],

    tdata_cache: Dict[float, TData],

    inherited_hi: Optional[float] = None,

    start_lower_hint: Optional[float] = None,

) -> dict:

    active = sorted(float(t) for t in active_t)

    tdata = [Get_Constraint_Data(num, ps, tdata_cache, t) for t in active]

    model = Build_With_r_Constraints(k, R_prev, ps, tdata)

    lo = float(R_prev)  # r 下界

    hi = float(num.S_k)  # r 上界

    if inherited_hi is not None:

        hi = min(hi, float(inherited_hi))  # 繼承 r 上界

    best_w: Optional[np.ndarray] = None

    best_z: Optional[np.ndarray] = None

    def evaluate_r(r: float):

        aval = max(0.0, num.S_k - r)

        bval = max(0.0, k * (r - R_prev))

        model.a.value = aval

        model.b.value = bval

        solve = Solve_With_r_Feasibility(

            k, R_prev, ps, model, aval, bval  # solve

        )

        if solve["classification"] != "feasible" or model.w.value is None:

            return solve, None, None

        wv = np.asarray(model.w.value, dtype=float).reshape(k)

        if model.z.value is None:

            return solve, None, None

        zv = np.asarray(model.z.value, dtype=float).reshape(k - 2)

        wv, zv = Clean_w_rho(wv, zv)  # 清理誤差

        return solve, wv, zv

    if start_lower_hint is not None:  #  start_lower_hint = S_k 截到小數後 2 位, 加速用可刪

        base_lo = float(lo)

        target = min(

            float(start_lower_hint),

            math.nextafter(float(hi), float(base_lo)),

        )

        if target > base_lo + 1e-14:

            for frac in (1.0, 0.75, 0.5, 0.25):

                seed = base_lo + frac * (target - base_lo)

                if not lo + 1e-14 < seed < hi - 1e-14:

                    continue

                seed_solve, seed_w, seed_z = evaluate_r(seed)

                cls = seed_solve["classification"]

                if cls == "feasible" and seed_w is not None and seed_z is not None:

                    lo = seed

                    best_w, best_z = seed_w.copy(), seed_z.copy()

                    break

                if cls == "infeasible":

                    hi = min(hi, seed)

    for _ in range(BISECTION_ITERS):

        if hi - lo <= 1e-14:

            break

        r = 0.5 * (lo + hi)  # 中點

        solve, wv, zv = evaluate_r(r)

        cls = solve["classification"]

        if cls == "feasible" and wv is not None and zv is not None:

            lo = r  # 可行 提高下界

            best_w, best_z = wv.copy(), zv.copy()

        elif cls == "infeasible":

            hi = r  # 不可行 降低上界

        else:

            break

    if best_w is None or best_z is None:

        raise RuntimeError(

            "no feasible trial above R_prev was found"

        )

    return {

        "hi": float(hi),

        "best_w": best_w,

        "best_z": best_z,

        "active_t": active,

    }

# ==============================================================================

# 7. NO-R_k fixed (w*,rho*,t) -> f and E

# ==============================================================================

def No_r_CVXPY_At_t(k: int, S_k: float, R_prev: float, ps: PrefixSystem, td: TData, w: np.ndarray, z: np.ndarray) -> dict:

    """NO-R_k: w,rho,t are fixed; CVXPY optimizes only f, then E is computed."""

    if abs(td.t) <= 1e-15:  # t = 0

        sw = s_w_numeric(k, w)

        bal = E_Balanced(k, S_k, R_prev, sw, 0.0)

        bounded = E_Bounded(k, S_k, R_prev, sw, 0.0)

        return {'t': 0.0, 'H': 0.0, 'delta': 0.0, 'balanced_exponent': float(bal['exponent']), 'balanced_gamma': float(bal['gamma']), 'gamma_valid': bool(bal['gamma_valid']), 'bounded_exponent': float(bounded['exponent']), 'bounded_gamma': float(bounded['gamma']), 'forced_error': 0.0, 'status': 'analytic'}

    v0 = td.A0 @ w

    v1 = td.A1 @ w

    special = td.special_factor_w1 * w[0]

    l0, l1 = Strengthened_Caps(ps, w, z, td.J_k)

    hat0, hat1 = (v0 - l0, v1 - l1)

    f0 = cp.Variable(ps.n_types, nonneg=True)

    f1 = cp.Variable(ps.n_types, nonneg=True)

    g0 = v0 - f0

    g1 = v1 - f1

    H = cp.sum(h_expr(g0, g1))

    constraints = [f0 <= hat0, f1 <= hat1, g0 >= 0.0, g1 >= 0.0, special + cp.sum(f0) + cp.sum(f1) == td.R_k]

    p = cp.Problem(cp.Maximize(H), constraints)  # 固定 w,rho,t 解 f

    status = Solve_No_r_CVXPY(p)

    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):

        raise RuntimeError(f'verification solve failed at t={td.t}: {status}')

    f0v = np.asarray(f0.value, dtype=float).reshape(ps.n_types)

    f1v = np.asarray(f1.value, dtype=float).reshape(ps.n_types)

    g0v = np.maximum(v0 - f0v, 0.0)

    g1v = np.maximum(v1 - f1v, 0.0)

    Hnum = sum((h_numeric(g0v[j], g1v[j]) for j in range(ps.n_types)))

    delta = td.t - td.R_k - Hnum  # exponent 改善量

    sw = s_w_numeric(k, w)

    bal = E_Balanced(k, S_k, R_prev, sw, delta)

    bounded = E_Bounded(k, S_k, R_prev, sw, delta)

    return {'t': float(td.t), 'H': float(Hnum), 'delta': float(delta), 'balanced_exponent': float(bal['exponent']), 'balanced_gamma': float(bal['gamma']), 'gamma_valid': bool(bal['gamma_valid']), 'bounded_exponent': float(bounded['exponent']), 'bounded_gamma': float(bounded['gamma']), 'forced_error': float(special + f0v.sum() + f1v.sum() - td.R_k), 'status': str(status)}

def Effective_E(row: dict) -> float:

    if row.get('gamma_valid', False) and math.isfinite(row.get('balanced_exponent', float('nan'))):  # 驗算 0 <= gamma <= 1/k

        return float(row['balanced_exponent'])

    return float(row['bounded_exponent'])

def Evaluate_E(k: int, num: KNumerics, R_prev: float, ps: PrefixSystem, w: np.ndarray, z: np.ndarray, t: float, tdata_cache: Dict[float, TData]) -> dict:

    td = Get_Constraint_Data(num, ps, tdata_cache, t)

    row = No_r_CVXPY_At_t(k, num.S_k, R_prev, ps, td, w, z)

    row['effective_exponent'] = Effective_E(row)

    return row

# ==============================================================================

# 8. NO-R_k dense t scan -> local minima -> golden search

# ==============================================================================

def Golden_Search(get_row, left: float, right: float, tol: float, max_iters: int) -> dict:

    """Numerically minimize the fixed-(w,rho) exponent on [left,right]."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0

    a0 = float(left)

    b0 = float(right)

    if b0 <= a0:

        return get_row(a0)

    c = b0 - (b0 - a0) / phi  # 左內點

    d = a0 + (b0 - a0) / phi  # 右內點

    rc = get_row(c)  # E(c)

    rd = get_row(d)  # E(d)

    for _ in range(max_iters):

        if b0 - a0 <= tol:

            break

        if rc['effective_exponent'] <= rd['effective_exponent']:

            b0, d, rd = (d, c, rc)

            c = b0 - (b0 - a0) / phi

            rc = get_row(c)

        else:

            a0, c, rc = (c, d, rd)

            d = a0 + (b0 - a0) / phi

            rd = get_row(d)

    candidates = [get_row(a0), get_row(b0), rc, rd, get_row(0.5 * (a0 + b0))]

    return min(candidates, key=lambda Q_k: Q_k['effective_exponent'])

def No_r_CVXPY(k: int, num: KNumerics, R_prev: float, ps: PrefixSystem, w: np.ndarray, z: np.ndarray, initial_grid: Sequence[float], active_t: Sequence[float], tdata_cache: Dict[float, TData], golden_tol: float, golden_iters: int, max_local_refinements: int) -> dict:

    row_cache: Dict[float, dict] = {}

    def get_row(t: float) -> dict:

        t = min(num.t_critical, max(0.0, float(t)))

        key = round(t, 14)

        if key not in row_cache:

            row_cache[key] = Evaluate_E(k, num, R_prev, ps, w, z, t, tdata_cache)

        return row_cache[key]

    scan = {0.0, float(num.t_critical), float(num.j_turn)}  # scan t

    for t in initial_grid:

        if 0.0 <= t <= num.t_critical:

            scan.add(float(t))

    for t in active_t:

        if 0.0 <= t <= num.t_critical:

            scan.add(float(t))

    xs = sorted(scan)

    rows: List[dict] = []

    for t in xs:

        row = get_row(t)    # step 4 : every t,w* find f

        rows.append(row)

    vals = [float(r['effective_exponent']) for r in rows]  # 每個 t 的 E

    ibest = int(np.argmin(vals))

    local: List[Tuple[float, float, float, float]] = []

    for i in range(1, len(xs) - 1):

        if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:

            local.append((vals[i], xs[i - 1], xs[i + 1], xs[i]))

    if 0 < ibest < len(xs) - 1:

        local.append((vals[ibest], xs[ibest - 1], xs[ibest + 1], xs[ibest]))

    local.sort(key=lambda Q_k: Q_k[0])

    unique: List[Tuple[float, float, float, float]] = []

    for item in local:

        center = item[3]    

        if all((abs(center - old[3]) > 1e-12 for old in unique)):

            unique.append(item)

        if len(unique) >= max_local_refinements:  # step 7 : get 3 local minima

            break

    candidates = [rows[ibest]]

    for _, left, right, center in unique:

        candidates.append(Golden_Search(get_row, left, right, golden_tol, golden_iters))    # step 8 : golden search -> get worst t

    best = min(candidates, key=lambda Q_k: Q_k['effective_exponent'])  # 找 worst t

    return {'t': float(best['t']), 'exponent': float(best['effective_exponent'])}

# ==============================================================================

# 9. Numerical Optimization

# ==============================================================================

def Initial_r_Hint(R_prev: float, S_k: float) -> float:

    h = math.floor(float(S_k) * 100.0) / 100.0

    h = max(float(R_prev), h)

    return min(h, math.nextafter(float(S_k), float(R_prev)))

def Merge_Thresholds(values: Sequence[float], tcrit: float) -> List[float]:

    vals = sorted((min(float(tcrit), max(0.0, float(x))) for x in values if float(x) > 0.0))

    out: List[float] = []

    for x in vals:

        if not out or abs(x - out[-1]) > 1e-12:

            out.append(x)

    if not out or abs(out[-1] - tcrit) > 1e-12:

        out.append(float(tcrit))

    return out

def Ceil_R_to_9_Decimals(x: float) -> float:

    return float(Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_CEILING))  # 小數後 9 位無條件進位

def Numerical_Optimization(k: int, R_prev: float) -> float:

    if k not in SUPPORTED_K:

        raise ValueError("k must be 3, 4, 5, 6, or 7")

    num = Build_K_Numerics(k)  # 建 k 的 PPSZ 數值

    ps = Build_Prefix_System(k)  # 建 prefix system

    scan_t = Build_Threshold_Set(num.t_critical, T_STEP)  # 0.005 t

    initial_t = Build_Threshold_Set(num.t_critical, ACTIVE_STEP)  # 0.05 t

    warm_extra = [float(num.j_turn)]  # 加入 j_turn

    initial_t = Merge_Thresholds(list(initial_t) + warm_extra, num.t_critical)

    start_lower_hint = Initial_r_Hint(R_prev, num.S_k)

    tdata_cache: Dict[float, TData] = {}

    active_t = list(initial_t)

    inherited_hi = float(num.S_k)

    final_with_r: Optional[dict] = None

    print(f"k = {k}")

    print(f"S_{k} = {num.S_k:.15f}")

    print(f"R_{k-1} = {R_prev:.15f}")

    while True:

        with_r_result = With_r_CVXPY(

            k, num, R_prev, ps, active_t, tdata_cache,

            inherited_hi=inherited_hi,

            start_lower_hint=start_lower_hint,

        )

        inherited_hi = min(inherited_hi, float(with_r_result["hi"]))

        w_star = with_r_result["best_w"]  # 最佳 w

        z_star = with_r_result["best_z"]  # 最佳 z=rho*w

        no_r_search = No_r_CVXPY(

            k, num, R_prev, ps, w_star, z_star,

            scan_t, active_t, tdata_cache,

            GOLDEN_TOL, GOLDEN_ITERS, MAX_LOCAL_REFINEMENTS,

        )

        t_star = float(no_r_search["t"])  # 最差 t

        e_star = float(no_r_search["exponent"])  # 最小 E

        nearest = min(abs(t_star - t) for t in active_t)  # 與 active t 最近距離

        print(f"E = {e_star:.12f}, t = {t_star:.12f}")

        final_with_r = with_r_result

        if t_star <= 1e-12 or nearest <= DUPLICATE_TOL:

            break

        active_t.append(t_star)  # 加入新 worst t

        active_t.sort()

    if final_with_r is None:

        raise RuntimeError("experiment produced no result")

    best_w = final_with_r["best_w"]

    best_z = final_with_r["best_z"]

    final_active_rows: List[dict] = []

    for t_i in final_with_r["active_t"]:

        row = Evaluate_E(k, num, R_prev, ps, best_w, best_z, t_i, tdata_cache)

        final_active_rows.append(row)

    if not final_active_rows:

        raise RuntimeError("final active t-set is empty")

    worst_row = min(final_active_rows, key=lambda row: float(row["effective_exponent"]))  # 最終最差 active t

    final_exp = float(worst_row["effective_exponent"])  # 最終 exponent

    final_t = float(worst_row["t"])

    if worst_row.get("gamma_valid", False):

        final_gamma = float(worst_row["balanced_gamma"])

    else:

        final_gamma = float(worst_row["bounded_gamma"])

    rhos = []

    for i in range(2, k):

        wi = float(best_w[i - 1])

        zi = float(best_z[i - 2])

        rhos.append(float("nan") if wi <= 1e-14 else zi / wi)

    print("\n" + "=" * 60)

    print(f"FINAL RESULT | k = {k}")

    print("=" * 60)

    print("w        = (" + ", ".join(f"{x:.12f}" for x in best_w) + ")")

    print("rho      = " + ", ".join(f"rho_{i}={rhos[i - 2]:.12f}" for i in range(2, k)))

    print(f"exponent = {final_exp:.12f}")

    print(f"base     = {2.0 ** final_exp:.12f}")

    print(f"argmin t = {final_t:.12f}")

    print(f"gamma    = {final_gamma:.12f}")

    return final_exp

def main() -> None:

    R_prev = 0.0  # R_2

    for k in SUPPORTED_K:

        final_exp = Numerical_Optimization(k, R_prev)  # 跑目前 k

        if k < SUPPORTED_K[-1]:

            R_prev = Ceil_R_to_9_Decimals(final_exp)  # 給下一個 k 的 R_(k-1)

            print(f"R_{k} used for k={k + 1} = {R_prev:.9f}")

            print("-" * 60)

if __name__ == "__main__":

    main()
