# %% [markdown]
# ## 10c — Base comum dos probes científicos (congelada)
# Todo probe devolve um `ProbeResult` com: estatística (p-valor, tamanho de efeito, IC bootstrap), controles
# negativos/positivos usados, sementes (de **holdout**: espaço de sementes nunca usado pela evolução), hash do genoma,
# hash das versões dos motores e dos thresholds. Probes são geradores: rendem entre blocos de ticks da arena.

# %%
SENSOR_RESOURCE = [1, 2, 3]
SENSOR_EXTERO = list(range(1, 23))
SENSOR_INTERO = [23, 24, 25, 26]
SENSOR_SOUND = [12, 13, 14, 15]
ACTION_PLACE, ACTION_PICK, ACTION_TAKE, ACTION_GIVE = 8, 7, 12, 11


@dataclass
class ProbeResult:
    level: str
    name: str
    passed: bool
    p: float
    effect: float
    ci: Tuple[float, float]
    n: int
    metrics: Dict[str, Any]
    controls: Dict[str, Any]
    seeds: List[int]
    genome_hash: str
    claim: str
    duration: float
    ts: str = field(default_factory=iso_now)
    versions: str = field(default_factory=lambda: ENGINE_VERSIONS_HASH)
    thresholds: str = ""
    notes: str = ""
    replay: Optional[Dict[str, Any]] = None

    def public(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "replay"}
        d["has_replay"] = self.replay is not None
        return to_jsonable(d)


def genome_hash(g: torch.Tensor) -> str:
    return hashlib.sha256(g.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()[:24]


def diff_ci(a, b, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, (n_boot, a.size))].mean(1) - b[rng.integers(0, b.size, (n_boot, b.size))].mean(1)
    return float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))


def compare_greater(a, b, th: Thresholds, min_effect: Optional[float] = None) -> Dict[str, Any]:
    """a > b? Mann–Whitney unilateral + d de Cohen + IC bootstrap da diferença de médias."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    p = mann_whitney_greater(a, b)
    d = cohens_d(a, b)
    ci = diff_ci(a, b)
    me = th.min_effect if min_effect is None else min_effect
    ok = bool(np.isfinite(p) and p < th.alpha and np.isfinite(d) and d >= me)
    return {"p": float(p) if np.isfinite(p) else 1.0, "d": float(d) if np.isfinite(d) else 0.0, "ci": ci,
            "mean_a": float(np.nanmean(a)) if a.size else float("nan"), "mean_b": float(np.nanmean(b)) if b.size else float("nan"),
            "passed": ok}


def memoryless(L: GenomeLayout, G: torch.Tensor) -> torch.Tensor:
    """Controle negativo de memória: sem recorrência, sem integração temporal (τ=1) e sem plasticidade."""
    G = G.clone()
    for name, val in [("ctl_W0", 0.0), ("ctl_logtau", 0.0), ("plast_eta", 0.0)]:
        a, b = L.slices[name]
        G[..., a:b] = val
    return L.clamp(G)


def passive_body(L: GenomeLayout, G: torch.Tensor) -> torch.Tensor:
    """Controle negativo de corpo: mesma morfologia, sem inervação motora (nenhuma atuação)."""
    G = G.clone()
    a, b = L.slices["body_innerv"]
    G[..., a:b] = 0.0
    return G


def probe_seeds(level: str, key: str, k: int, holdout: bool = True) -> List[int]:
    space = "holdout" if holdout else "screen"
    return [derive_seed(space, level, key, i) for i in range(k)]


def sensor_mask(n: int, idx: Sequence[int], device) -> torch.Tensor:
    m = torch.ones(n, N_SENSORS, device=device)
    m[:, list(idx)] = 0.0
    return m


def make_replay(trial: Trial, rec: Dict[str, torch.Tensor], k: int = 8, every: int = 1) -> Dict[str, Any]:
    """Replay científico compacto (derivado da execução real): trajetórias das primeiras arenas + layout."""
    pos = rec.get("pos")
    if pos is None or pos.numel() == 0:
        return {}
    k = min(k, trial.n)
    arenas = sorted(set(trial.arena_of[:k].tolist()))
    S = trial.ws.arena["size"] if trial.ws.arena else 16
    lay = []
    for a in arenas:
        oy, ox = trial.ws.arena["origins"][a]
        sub = lambda t: t[oy:oy + S, ox:ox + S].cpu().numpy()
        lay.append({"arena": a, "origin": [int(ox), int(oy)], "water": np.nonzero(sub(trial.ws.is_water & ~trial.ws.filled)),
                    "filled": np.nonzero(sub(trial.ws.filled)), "struct": np.nonzero(sub(trial.ws.struct_h > 0)),
                    "rich": np.nonzero(sub(trial.ws.rich)), "patches": trial.patch_tiles[a].cpu().numpy().tolist()})
    P = pos[::every, :k].cpu().numpy().astype(np.float32)
    org = np.asarray([trial.ws.arena["origins"][int(a)] for a in trial.arena_of[:k].tolist()], np.float32)
    P = P - org[None, :, ::-1]                                    # coordenadas locais da arena (x, y)
    return {"size": int(S), "pos": P.astype(np.float16), "arena_of": trial.arena_of[:k].cpu().numpy(),
            "layout": [{kk: (np.stack(v, 1).astype(np.int16).tolist() if isinstance(v, tuple) else v) for kk, v in L_.items()}
                       for L_ in lay], "ticks": int(P.shape[0]), "every": every}


def result(level: str, name: str, passed: bool, stat: Dict[str, Any], n: int, metrics: Dict[str, Any],
           controls: Dict[str, Any], seeds: List[int], G: torch.Tensor, t0: float, th_hash: str,
           claim_pass: str = "TESTED", notes: str = "", replay=None) -> ProbeResult:
    return ProbeResult(level=level, name=name, passed=bool(passed), p=float(stat.get("p", 1.0)),
                       effect=float(stat.get("d", 0.0)), ci=tuple(stat.get("ci", (float("nan"), float("nan")))), n=int(n),
                       metrics=to_jsonable(metrics), controls=to_jsonable(controls), seeds=list(seeds),
                       genome_hash=genome_hash(G), claim=claim_pass if passed else "NOT YET VALIDATED",
                       duration=time.time() - t0, thresholds=th_hash, notes=notes, replay=replay)


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """Teste exato de Fisher unilateral: proporção 1 > proporção 2 (soma hipergeométrica)."""
    K, N = k1 + k2, n1 + n2
    if n1 == 0 or n2 == 0:
        return 1.0
    def hyp(k):
        return math.comb(K, k) * math.comb(N - K, n1 - k) / math.comb(N, n1)
    return float(min(1.0, sum(hyp(k) for k in range(k1, min(K, n1) + 1))))


def ridge_r2_cv(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lam: float = 1.0) -> float:
    """R² médio (validação cruzada por grupo = arena; sem vazamento temporal entre treino e teste)."""
    ug = np.unique(groups)
    k = min(4, len(ug))
    if k < 2:
        return 0.0
    folds = {g: i % k for i, g in enumerate(ug)}
    f = np.asarray([folds[g] for g in groups])
    pred = np.zeros_like(Y)
    for i in range(k):
        tr, te = f != i, f == i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xa = np.hstack([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))]); Xb = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))])
        Wt = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Y[tr])
        pred[te] = Xb @ Wt
    ss_res = ((Y - pred) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float(np.mean(1 - ss_res / ss_tot))
