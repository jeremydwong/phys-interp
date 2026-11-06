"""Reusable neuron-cleanliness measurements (model- and domain-agnostic; numpy + optional torch).

Written for the vis_ ladder study and meant to be ported to other models (e.g. Pluto). Every function
takes plain arrays: per-example pole scores S [N, U] (N examples, U units) and ground-truth factors
{name: values [N]} with a kind "cat" (categorical) or "cont" (continuous). Nothing here knows about images.

Definitions (repo conventions from CLAUDE.md, plus our formalizations of the essay's prose, marked OURS):

poles(z)            For a unit with a spatial / token axis, the positive pole score of an example is the max
                    of its PRE-activation over positions, the negative pole score is -min (so both poles are
                    "bigger = stronger"). Essay: "Each neuron's 20 strongest responses out of 50,000 images";
                    the negative pole is the argmin set. Pre-activation because post-ReLU hides the negative side.
eta2(score, codes)  Correlation ratio: share of the score's variance explained by group means over the codes.
                    For a continuous factor the codes are 30 quantile bins of the factor, which makes this
                    identical to analysis.common.binned_r2(factor, score) (same quantile-bin construction).
census(...)         Per unit-pole: best single factor R^2; "clean" >= 0.5, "partial" 0.25-0.5, else
                    "unexplained" (repo convention); "conjunction" if the best PAIR of factors (joint codes:
                    categorical x categorical, continuous -> 6 quantile bins) beats the best single by > 0.15.
purity(...)         OURS. "One main job" on strong activations: take the pole's strong set (top 1% of examples
                    by percentile, or, raw-shape version, examples above half the pole's raw range, clipped to
                    [20, 10%]); for each factor (continuous ones cut into quintiles) the modal value's share p,
                    chance-corrected against the factor's largest marginal share p0: (p - p0) / (1 - p0).
                    A pole has one main job if the best corrected purity >= 0.5.
dial_detector(...)  OURS. From the raw activation shape: r = (s - median) / (max - median). A clean pole is a
                    "dial" if its best factor still explains >= half as much variance among the weak examples
                    (0 < r <= 0.5) as overall, else a "detector" (meaning lives only near the top).
weak_strong_thresholds(...)  OURS (essay: "removing the weakest 75%" vs "the strongest 7%" at equal total
                    activation). Per unit: tau_weak = 75th percentile of |a| over its nonzero activations;
                    tau_strong chosen so the activation mass above it equals the mass below tau_weak.
TopKSAE / near_copy_share(...)  OURS formalization of "88% of its activity is near-copies of single neurons":
                    decoder direction d_f is a near copy if max_i |cos(d_f, e_i)| >= 0.9; share = sum of mean
                    activation of near-copy features / sum over all features.
rotation_search(...) OURS formalization (decaying-step schedule, identity-start sanity check) of "search the rotations of a layer for the one where a handful of
                    units carry the most activity, starting from a random one": maximise E sum_i (R x)_i^4 over
                    orthogonal R from a random start (data centred and scaled globally, which is basis-free);
                    report how close the rotated axes land to the neuron axes.
"""
from __future__ import annotations

import numpy as np

CLEAN, PARTIAL, CONJ = 0.5, 0.25, 0.15
N_BINS, PAIR_BINS = 30, 6


# ------------------------------------------------------------------ poles

def poles(z):
    """z: [B, C, ...positions] pre-activation (numpy or torch) -> (pos [B, C], neg [B, C]), neg = -min.

    2-D input [B, C] (no position axis): pos = max(z, 0), neg = max(-z, 0). (Fixed 2026-11: this used to return
    (z, -z), which gives the two poles identical rank orders up to reversal and so identical census R^2.)"""
    if z.ndim == 2:
        if hasattr(z, "clamp"):
            return z.clamp(min=0), (-z).clamp(min=0)
        return np.maximum(z, 0), np.maximum(-z, 0)
    flat = z.reshape(z.shape[0], z.shape[1], -1)
    if hasattr(flat, "amax"):
        return flat.amax(-1), -flat.amin(-1)
    return flat.max(-1), -flat.min(-1)


# ------------------------------------------------------------------ factor codes and eta^2

def quantile_codes(x, bins):
    x = np.asarray(x, np.float64)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)


def factor_codes(factors: dict, kinds: dict, bins=N_BINS):
    """-> {name: int codes [N]}: categorical values relabelled 0..K-1, continuous -> quantile bins."""
    out = {}
    for k, v in factors.items():
        if kinds[k] == "cat":
            out[k] = np.unique(np.asarray(v), return_inverse=True)[1]
        else:
            out[k] = quantile_codes(v, bins)
    return out


def _dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _centered(S):
    import torch
    S = torch.as_tensor(np.asarray(S), dtype=torch.float32, device=_dev())
    S = S - S.mean(0)
    return S, (S ** 2).sum(0) + 1e-12


def _group_sums(Sc, codes, K=None):
    import torch
    codes = torch.as_tensor(np.asarray(codes), device=Sc.device).long()
    K = int(codes.max()) + 1 if K is None else K
    sums = torch.zeros(K, Sc.shape[1], device=Sc.device).index_add_(0, codes, Sc)
    cnt = torch.bincount(codes, minlength=K).float()
    return sums, cnt


def eta2(S, codes):
    """S [N, U] scores, codes [N] ints -> [U] correlation ratio (between-group / total variance).
    With codes = 30 quantile bins of a continuous factor this equals analysis.common.binned_r2(factor, score)."""
    Sc, tot = _centered(S)
    sums, cnt = _group_sums(Sc, codes)
    return ((sums ** 2 / cnt.clamp(min=1)[:, None]).sum(0) / tot).cpu().numpy()


def _indicator_r2(sums, cnt, N, tot):
    """R^2 of every one-vs-rest indicator 1[code = v] (2-group eta^2): [K, U]."""
    n = cnt[:, None]
    return sums ** 2 * (1 / n.clamp(min=1) + 1 / (N - n).clamp(min=1)) / tot


def census(S, factors: dict, kinds: dict, pair_factors: list | None = None):
    """Census of one pole for all units. S [N, U] pole scores.

    Single-factor R^2: continuous factor -> binned R^2 (30 quantile bins; any 1-D tuning curve);
    categorical factor -> the best one-vs-rest indicator R^2 over its values ("the unit detects value v").
    The indicator rule (OURS) keeps a high-cardinality factor such as the class label from explaining an
    arbitrary mixture of classes as if it were one job. Binary derived factors (curved, warm, textured) let a
    unit that detects a natural grouping of values count as clean.
    Pair R^2 (conjunction test) over the base factors: cat x cat -> best 2x2 table of two indicators;
    cat x cont -> indicator x 6 quantile bins; cont x cont -> 6 x 6 bins.
    Returns dict: r2 [U, F], names, best, best_r2, best_name, best_value (categorical value index or -1),
    pair_r2, pair, conj, label."""
    import torch
    names = list(factors)
    Sc, tot = _centered(S)
    N = Sc.shape[0]
    r2, bval = [], []
    for k in names:
        if kinds[k] == "cat":
            codes = np.unique(np.asarray(factors[k]), return_inverse=True)[1]
            sums, cnt = _group_sums(Sc, codes)
            ir = _indicator_r2(sums, cnt, N, tot)
            m, a = ir.max(0)
            r2.append(m)
            bval.append(a)
        else:
            sums, cnt = _group_sums(Sc, quantile_codes(factors[k], N_BINS))
            r2.append((sums ** 2 / cnt.clamp(min=1)[:, None]).sum(0) / tot)
            bval.append(torch.full_like(r2[-1], -1).long())
    r2 = torch.stack(r2, 1)
    best_r2, best = r2.max(1)
    best_value = torch.stack(bval, 1).gather(1, best[:, None])[:, 0]
    pf = pair_factors if pair_factors is not None else list(names)
    pair_r2 = torch.zeros(Sc.shape[1], device=Sc.device)
    pair_id = torch.full((Sc.shape[1],), -1, device=Sc.device).long()
    pnames = []
    codes = {k: (np.unique(np.asarray(factors[k]), return_inverse=True)[1] if kinds[k] == "cat"
                 else quantile_codes(factors[k], PAIR_BINS)) for k in pf}
    for i in range(len(pf)):
        for j in range(i + 1, len(pf)):
            a, b = pf[i], pf[j]
            if kinds[a] == "cont" and kinds[b] == "cat":
                a, b = b, a
            Ka, Kb = int(codes[a].max()) + 1, int(codes[b].max()) + 1
            J, nJ = _group_sums(Sc, codes[a] * Kb + codes[b], Ka * Kb)
            J, nJ = J.view(Ka, Kb, -1), nJ.view(Ka, Kb, 1)
            A, nA = J.sum(1), nJ.sum(1)                    # [Ka, U], [Ka, 1]
            B, nB = J.sum(0), nJ.sum(0)                    # [Kb, U], [Kb, 1]
            if kinds[a] == "cat" and kinds[b] == "cat":    # best 2x2 table over (v1, v2)
                s11, n11 = J, nJ
                s10, n10 = A[:, None] - J, nA[:, None] - nJ
                s01, n01 = B[None] - J, nB[None] - nJ
                s00, n00 = -(s11 + s10 + s01), N - n11 - n10 - n01
                bt = sum(s ** 2 / n.clamp(min=1) for s, n in ((s11, n11), (s10, n10), (s01, n01), (s00, n00)))
                e = (bt / tot).flatten(0, 1).max(0).values
            elif kinds[a] == "cat":                         # indicator(a = v1) x bins(b)
                rest, nrest = B[None] - J, nB[None] - nJ
                bt = (J ** 2 / nJ.clamp(min=1)).sum(1) + (rest ** 2 / nrest.clamp(min=1)).sum(1)
                e = (bt / tot).max(0).values
            else:                                           # 6 x 6 joint bins
                e = (J ** 2 / nJ.clamp(min=1)).sum((0, 1)) / tot
            upd = e > pair_r2
            pair_r2 = torch.where(upd, e, pair_r2)
            pair_id = torch.where(upd, torch.full_like(pair_id, len(pnames)), pair_id)
            pnames.append(f"{pf[i]}x{pf[j]}")
    r2, best, best_r2, best_value = r2.cpu().numpy(), best.cpu().numpy(), best_r2.cpu().numpy(), best_value.cpu().numpy()
    pair_r2, pair_id = pair_r2.cpu().numpy(), pair_id.cpu().numpy()
    pair = np.array([pnames[p] if p >= 0 else "" for p in pair_id], dtype=object)
    label = np.where(best_r2 >= CLEAN, "clean", np.where(best_r2 >= PARTIAL, "partial", "unexplained"))
    return dict(r2=r2, names=names, best=best, best_r2=best_r2, best_name=np.array(names)[best], best_value=best_value,
                pair_r2=pair_r2, pair=pair, conj=pair_r2 > best_r2 + CONJ, label=label)


def census_summary(c):
    lab = c["label"]
    return dict(clean=float((lab == "clean").mean()), partial=float((lab == "partial").mean()),
                unexplained=float((lab == "unexplained").mean()), conj=float(c["conj"].mean()),
                mean_best_r2=float(c["best_r2"].mean()), n=int(len(lab)))


# ------------------------------------------------------------------ strong-activation purity ("one main job")

def strong_set(s, mode="pct", top_frac=0.01, raw_frac=0.5, lo=20, hi_frac=0.10):
    """Indices of a pole's strong examples. mode "pct": top `top_frac` by rank (percentile view).
    mode "raw": examples with (s - median) >= raw_frac * (max - median), clipped to [lo, hi_frac*N] (OURS)."""
    N = len(s)
    order = np.argsort(-s)
    if mode == "pct":
        return order[:max(1, int(round(top_frac * N)))]
    med = np.median(s)
    n = int(((s - med) >= raw_frac * (s.max() - med)).sum())
    return order[:int(np.clip(n, lo, int(hi_frac * N)))]


def purity(S, factors: dict, kinds: dict, mode="pct", cat_only=True, **kw):
    """Per unit: best chance-corrected purity over factors, the factor, its modal value, and set size.

    cat_only (default True, fixed 2026-11): take the best over CATEGORICAL factors only. Taking the best over
    every factor, with continuous ones cut into quintiles, saturates to ~100% "one job" even for random-init
    nets when there are many continuous factors (some quintile of some nuisance is always over-represented in a
    small strong set). If no factor is categorical, falls back to all factors."""
    names = [k for k in factors if kinds[k] == "cat"] if cat_only else list(factors)
    if not names:
        names = list(factors)
    factors = {k: factors[k] for k in names}
    codes = {k: (np.unique(np.asarray(v), return_inverse=True)[1] if kinds[k] == "cat" else quantile_codes(v, 5))
             for k, v in factors.items()}
    p0 = {k: np.bincount(c).max() / len(c) for k, c in codes.items()}
    U = S.shape[1]
    best = np.zeros(U)
    best_f = np.empty(U, dtype=object)
    best_v = np.zeros(U, int)
    nset = np.zeros(U, int)
    for u in range(U):
        idx = strong_set(S[:, u], mode, **kw)
        nset[u] = len(idx)
        for k in names:
            bc = np.bincount(codes[k][idx], minlength=codes[k].max() + 1)
            p = bc.max() / len(idx)
            q = (p - p0[k]) / (1 - p0[k])
            if q > best[u] or best_f[u] is None:
                best[u], best_f[u], best_v[u] = q, k, int(bc.argmax())
    return dict(purity=best, factor=best_f, value=best_v, n=nset, one_job=best >= 0.5)


# ------------------------------------------------------------------ raw shape: dial vs detector

def dial_detector(s, y, kind, value=None):
    """OURS. s [N] pole score, y [N] the pole's best factor (for a categorical factor, `value` is the detected
    value's index and the factor is reduced to that indicator). Returns (is_dial, weak_r2, full_r2, frac_above_half)."""
    if kind == "cat":
        codes = np.unique(np.asarray(y), return_inverse=True)[1]
        if value is not None and value >= 0:
            codes = (codes == value).astype(int)
    else:
        codes = quantile_codes(y, N_BINS)
    full = float(eta2(s[:, None], codes)[0])
    med = np.median(s)
    r = (s - med) / (s.max() - med + 1e-12)
    weak = (r > 0) & (r <= 0.5)
    if weak.sum() < 50:
        return False, 0.0, full, float((r > 0.5).mean())
    wc = np.unique(codes[weak], return_inverse=True)[1]
    wr2 = float(eta2(s[weak][:, None], wc)[0]) if wc.max() > 0 else 0.0
    return wr2 >= 0.5 * full, wr2, full, float((r > 0.5).mean())


def rank_transform(S):
    """Percentile view of scores (each unit's empirical CDF), for the raw-vs-percentile comparison."""
    return np.argsort(np.argsort(S, 0), 0).astype(np.float64) / (len(S) - 1)


# ------------------------------------------------------------------ weak vs strong activation removal

def weak_strong_thresholds(A, weak_frac=0.75):
    """A [n, U] post-activation samples (numpy or torch). Per unit: tau_weak (|a| quantile over nonzero entries),
    tau_strong (mass-matched), and the fraction of nonzero entries above tau_strong. OURS."""
    import torch
    A = torch.as_tensor(A).float().abs()
    U = A.shape[1]
    tw, ts, fs = torch.zeros(U), torch.zeros(U), torch.zeros(U)
    for u in range(U):
        a = A[:, u]
        a = a[a > 1e-6]
        if len(a) < 10:
            tw[u], ts[u] = float("inf"), float("inf")
            continue
        a, _ = torch.sort(a)
        k = int(weak_frac * len(a))
        tw[u] = a[k]
        mass_weak = a[:k].sum()
        cs = torch.flip(torch.cumsum(torch.flip(a, [0]), 0), [0])          # mass at or above each index
        j = int(torch.nonzero(cs >= mass_weak).max()) if (cs >= mass_weak).any() else 0
        ts[u] = a[j]
        fs[u] = (len(a) - j) / len(a)
    return tw, ts, fs


def mask_fn(tau, mode):
    """Hook for model.edit: zero |a| < tau ("weak") or |a| >= tau ("strong"). tau [C] on the right device."""
    def f(a, name):
        t = tau.view(1, -1, *([1] * (a.ndim - 2))).to(a.dtype)
        keep = (a.abs() >= t) if mode == "weak" else (a.abs() < t)
        return a * keep
    return f


# ------------------------------------------------------------------ SAE near-copy share

def train_topk_sae(X, n_feat, k, steps=3000, bs=2048, lr=1e-3, seed=0, device="cuda"):
    """TopK SAE (Gao et al. 2024) without the aux loss. X [n, d] (torch). Returns dict with W_enc, b_enc, W_dec
    (unit-norm rows, [n_feat, d]), b_pre, fve (held-out fraction of variance explained)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = X.to(device).float()
    n, d = X.shape
    perm = torch.randperm(n, generator=g).to(device)
    Xtr, Xte = X[perm[: int(0.9 * n)]], X[perm[int(0.9 * n):]]
    b_pre = Xtr.mean(0).clone().requires_grad_(True)
    W_dec = torch.randn(n_feat, d, generator=g).to(device)
    W_dec = (W_dec / W_dec.norm(dim=1, keepdim=True)).requires_grad_(True)
    W_enc = W_dec.detach().T.clone().requires_grad_(True)
    b_enc = torch.zeros(n_feat, device=device, requires_grad=True)
    opt = torch.optim.Adam([b_pre, W_dec, W_enc, b_enc], lr=lr)

    def encode(x):
        h = (x - b_pre) @ W_enc + b_enc
        v, i = h.topk(k, dim=1)
        return torch.zeros_like(h).scatter_(1, i, torch.relu(v))

    for s in range(steps):
        idx = torch.randint(0, len(Xtr), (bs,), generator=g).to(device)
        x = Xtr[idx]
        z = encode(x)
        loss = ((z @ W_dec + b_pre - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True)
    with torch.no_grad():
        z = encode(Xte)
        rec = z @ W_dec + b_pre
        fve = 1 - ((rec - Xte) ** 2).sum() / ((Xte - Xte.mean(0)) ** 2).sum()
        zsum = torch.zeros(n_feat, device=device)
        for i in range(0, n, 8192):         # running sum (a dense [n, n_feat] code matrix does not fit for big layers)
            zsum += encode(X[i:i + 8192]).sum(0)
    return dict(W_enc=W_enc.detach(), b_enc=b_enc.detach(), W_dec=W_dec.detach(), b_pre=b_pre.detach(),
                fve=float(fve), mean_z=(zsum / n).cpu().numpy(), k=k, encode=encode)


def near_copy_share(W_dec, mean_z, basis=None, kappa=0.9):
    """W_dec [F, d] decoder rows; basis [d, d] rows = the 'neuron' axes (default identity).
    Returns (share of activity in near-copy features, fraction of live features that are near copies, c_f)."""
    import torch
    W = torch.as_tensor(W_dec).float()
    W = W / W.norm(dim=1, keepdim=True)
    if basis is not None:
        W = W @ torch.as_tensor(basis).float().T.to(W.device)
    c = W.abs().max(1).values.cpu().numpy()
    m = np.asarray(mean_z)
    live = m > 0
    share = float(m[c >= kappa].sum() / (m.sum() + 1e-12))
    return share, float((c[live] >= kappa).mean()) if live.any() else 0.0, c


# ------------------------------------------------------------------ rotation search

def random_rotation(d, seed=0):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def _rotation_ascent(X, R, steps, lr, device):
    """Normalised Riemannian gradient ascent on O(d) with QR retraction; returns (R_best, obj_best)."""
    import torch
    d = X.shape[1]
    eye = torch.eye(d, device=device)
    best_R, best_f = R.clone(), -float("inf")
    for s in range(steps):
        R.requires_grad_(True)
        f = (X @ R.T).pow(4).sum(1).mean()
        G, = torch.autograd.grad(f, R)
        with torch.no_grad():
            if float(f) > best_f:
                best_f, best_R = float(f), R.detach().clone()
            A = G @ R.T
            A = A - A.T                       # skew part = Riemannian gradient direction
            A = A / (A.norm() + 1e-12)
            Q, Rr = torch.linalg.qr((eye + lr * A) @ R.detach())
            R = (Q * torch.sign(torch.diagonal(Rr))[None]).detach()
    with torch.no_grad():
        f = float((X @ R.T).pow(4).sum(1).mean())
    if f > best_f:
        best_f, best_R = f, R
    return best_R, best_f


def rotation_search(X, steps=1000, lr=None, seed=0, device="cuda", init=None, schedule=(1, 4, 16, 64),
                    check_identity=True):
    """OURS. X [n, d] activations. Maximise mean_x sum_i (x R^T)_i^4 over orthogonal R (normalised Riemannian
    gradient ascent with QR retraction), starting from a random rotation (or `init`).

    Fixed 2026-11: the step is decayed in stages lr0 / f for f in `schedule` (default x1, 1/4, 1/16, 1/64), each
    stage `steps // len(schedule)` steps starting from the previous stage's best R; the best R seen is kept. With
    the old fixed step (lr0 = 0.5 sqrt(d/256), 0.71 at d = 512) the search oscillated and drifted away from the
    optimum even when started at the identity. With check_identity, the same schedule is also run from the
    identity (the neuron basis) as a sanity check: a working optimiser started at a local optimum stays there.

    Returns R [d, d] (rows = found axes), align = per found axis max_i |cos(R_k, e_i)|, objective values
    (start, end, neuron basis), per-stage history, and (check_identity) obj_identity_end / align_identity."""
    import torch
    X = torch.as_tensor(X).float().to(device)
    X = X - X.mean(0)
    X = X / X.pow(2).sum(1).mean().sqrt()
    d = X.shape[1]
    lr0 = 0.5 * (d / 256) ** 0.5 if lr is None else lr
    n_st = max(1, steps // len(schedule))

    def obj(R):
        with torch.no_grad():
            return float((X @ R.T).pow(4).sum(1).mean())

    def run(R):
        hist = []
        for f in schedule:
            R, o = _rotation_ascent(X, R, n_st, lr0 / f, device)
            hist.append(dict(lr=lr0 / f, obj=o, align_mean=float(R.abs().max(1).values.mean())))
        return R, hist

    R0 = torch.as_tensor(random_rotation(d, seed) if init is None else init).float().to(device)
    o_start, o_neuron = obj(R0), obj(torch.eye(d, device=device))
    R, hist = run(R0)
    align = R.abs().max(1).values.cpu().numpy()
    out = dict(R=R.cpu().numpy(), align=align, obj_start=o_start, obj_end=obj(R), obj_neuron=o_neuron, history=hist)
    if check_identity:
        Ri, _ = run(torch.eye(d, device=device))
        ai = Ri.abs().max(1).values.cpu().numpy()
        out.update(obj_identity_end=obj(Ri), align_identity=ai, align_identity_mean=float(ai.mean()),
                   identity_aligned_0p9=float((ai >= 0.9).mean()))
    return out
