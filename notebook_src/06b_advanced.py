# %% [markdown]
# ### 6b. Análise avançada (somente candidatos fortes / ChampionLab)
# Instrumentos externos de caracterização: transfer entropy aproximada e teste tipo Granger entre regiões
# (correlação preditiva **não** é causalidade física — a intervenção tem prioridade), lei efetiva macroscópica por
# coarse-graining (AR linear em blocos), variáveis emergentes (PCA de canais), recorrência (atratores / ciclos-limite),
# rastreamento de objetos (tempo de vida, trajetórias), hierarquia multiescala, varredura de fase (transições abruptas),
# histerese ambiental, localização da memória e mapa de importância causal por dano em blocos.

# %%
def _transfer_entropy(x: np.ndarray, y: np.ndarray, bins: int = 3) -> float:
    """TE_{x->y} com histórico 1, discretização por quantis (estimador plug-in; viesado para séries curtas)."""
    if len(x) < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    q = lambda s: np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, bins + 1)[1:-1]), s), 0, bins - 1)
    xd, yd = q(x), q(y)
    y1, y0, x0 = yd[1:], yd[:-1], xd[:-1]
    joint = np.zeros((bins, bins, bins))
    for a, b, c in zip(y1, y0, x0):
        joint[a, b, c] += 1
    joint /= joint.sum()
    p_y0x0 = joint.sum(0); p_y1y0 = joint.sum(2); p_y0 = joint.sum((0, 2))
    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for c in range(bins):
                p = joint[a, b, c]
                if p > 0 and p_y0x0[b, c] > 0 and p_y1y0[a, b] > 0 and p_y0[b] > 0:
                    te += p * math.log2(p * p_y0[b] / (p_y0x0[b, c] * p_y1y0[a, b]))
    return float(max(0.0, te))


def _granger_gain(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return 0.0
    Y = y[1:]
    A1 = np.stack([y[:-1], np.ones(len(Y))], 1)
    A2 = np.stack([y[:-1], x[:-1], np.ones(len(Y))], 1)
    r1 = Y - A1 @ np.linalg.lstsq(A1, Y, rcond=None)[0]
    r2 = Y - A2 @ np.linalg.lstsq(A2, Y, rcond=None)[0]
    return float(max(0.0, 1 - (r2 @ r2) / (r1 @ r1 + 1e-12)))


def probe_advanced(self: ValidationEngine, genome: Genome, seed: int, split: str = "holdout", steps: int = 256) -> TestResult:
    out: Dict[str, Any] = {}
    n = 4
    gb, st, prot = self._setup(genome, n, split, seed)
    S = gb.S
    m = min(self.cfg.metric_res, S)
    d, (ay, ax), (by, bx) = self._channel_geometry(S)
    r = self.ps.obs_radius
    st = self.sim.run(gb, st, self.ps.warmup)
    frames, sa, sb = [], [], []
    for t in range(steps):
        st = self.sim.step(gb, st)
        U = st.U
        sa.append(U[:, :, ay - r:ay + r + 1, ax - r:ax + r + 1].sum(1).mean((1, 2)).cpu().numpy())
        sb.append(U[:, :, by - r:by + r + 1, bx - r:bx + r + 1].sum(1).mean((1, 2)).cpu().numpy())
        if t % 2 == 1:
            frames.append(F.adaptive_avg_pool2d(U, m).cpu().numpy())
    sa, sb = np.asarray(sa), np.asarray(sb)
    Fr = np.asarray(frames)                       # [T, n, C, m, m]
    out["transfer_entropy_A_to_B"] = float(np.mean([_transfer_entropy(sa[:, i], sb[:, i]) for i in range(n)]))
    out["transfer_entropy_B_to_A"] = float(np.mean([_transfer_entropy(sb[:, i], sa[:, i]) for i in range(n)]))
    out["granger_gain_A_to_B"] = float(np.mean([_granger_gain(sa[:, i], sb[:, i]) for i in range(n)]))
    out["granger_gain_B_to_A"] = float(np.mean([_granger_gain(sb[:, i], sa[:, i]) for i in range(n)]))
    # lei efetiva por coarse-graining (blocos 4x4) — AR(1) linear com ridge, R² fora da amostra vs persistência
    T = Fr.shape[0]
    blocks = max(2, m // 8)
    Xc = np.stack([F.adaptive_avg_pool2d(torch.as_tensor(Fr[:, i]), blocks).flatten(1).numpy() for i in range(n)])
    tr = int(T * 0.7)
    r2s, r2p = [], []
    for i in range(n):
        X0, X1 = Xc[i, :-1], Xc[i, 1:]
        A = np.hstack([X0[:tr], np.ones((tr, 1))])
        Wt = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ X1[:tr])
        pred = np.hstack([X0[tr:], np.ones((len(X0) - tr, 1))]) @ Wt
        ss = ((X1[tr:] - X1[tr:].mean(0)) ** 2).sum() + 1e-12
        r2s.append(1 - ((X1[tr:] - pred) ** 2).sum() / ss)
        r2p.append(1 - ((X1[tr:] - X0[tr:]) ** 2).sum() / ss)
    out["effective_law_R2"] = float(np.mean(r2s)); out["persistence_baseline_R2"] = float(np.mean(r2p))
    out["macroscopic_law_detected"] = bool(np.mean(r2s) > max(0.5, np.mean(r2p) + 0.05))
    # variáveis emergentes (PCA de canais)
    Xch = Fr.transpose(0, 1, 3, 4, 2).reshape(-1, Fr.shape[2])
    Xch = Xch - Xch.mean(0)
    try:
        sv = np.linalg.svd(Xch[:: max(1, len(Xch) // 20000)], compute_uv=False)
        ev = (sv ** 2) / ((sv ** 2).sum() + 1e-12)
    except Exception:
        ev = np.zeros(Fr.shape[2])
    out["channel_pca_explained"] = ev[:4].round(4).tolist()
    # recorrência: distância mínima a estados anteriores (atrator fixo / ciclo-limite / caos)
    Z = Fr[:, 0].reshape(T, -1)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    late = Zn[T // 2:]
    D = 1 - late @ Zn[: T // 2].T
    rec = float(D.min(1).mean())
    lag_best = int(np.argmax([np.mean(np.sum(Zn[k:] * Zn[:-k], 1)) for k in range(2, max(3, T // 3))]) + 2) if T > 6 else 0
    out["recurrence_distance"] = rec
    out["attractor_type"] = ("fixed_point" if np.mean(np.abs(np.diff(Z[-10:], axis=0))) < 1e-5 else
                             ("limit_cycle_candidate" if rec < 0.01 else "aperiodic"))
    out["dominant_period_frames"] = lag_best
    # rastreamento de objetos
    lifetimes, speeds = [], []
    if HAVE_SCIPY:
        prev = None
        alive: Dict[int, Dict[str, Any]] = {}
        nid = 0
        for t in range(T):
            mag = np.sqrt((Fr[t, 0] ** 2).sum(0))
            mask = mag > mag.mean() + mag.std()
            lab, nl = sndi.label(mask)
            cents = sndi.center_of_mass(mask, lab, range(1, nl + 1)) if nl else []
            new_alive = {}
            for c in cents:
                best, bd = None, 3.0
                for k_, o in alive.items():
                    dd = math.hypot(c[0] - o["pos"][0], c[1] - o["pos"][1])
                    if dd < bd and k_ not in new_alive:
                        best, bd = k_, dd
                if best is None:
                    new_alive[nid] = {"pos": c, "born": t, "path": [c]}; nid += 1
                else:
                    o = alive[best]; o["path"].append(c); o["pos"] = c; new_alive[best] = o
            for k_, o in alive.items():
                if k_ not in new_alive:
                    lifetimes.append(t - o["born"])
                    if len(o["path"]) > 2:
                        p = np.asarray(o["path"]); speeds.append(float(np.linalg.norm(p[-1] - p[0]) / len(p)))
            alive = new_alive
        for o in alive.values():
            lifetimes.append(T - o["born"])
    out["object_lifetimes"] = summary_stats(lifetimes) if lifetimes else {"n": 0}
    out["object_speed_cells_per_frame"] = summary_stats(speeds) if speeds else {"n": 0}
    # hierarquia multiescala: estruturas em resoluções m, m/2, m/4
    hier = {}
    last = torch.as_tensor(Fr[-1])
    for res_ in [m, max(4, m // 2), max(2, m // 4)]:
        mg = F.adaptive_avg_pool2d(last, res_).pow(2).sum(1).sqrt()
        mk = mg > mg.mean((1, 2), keepdim=True) + mg.std((1, 2), keepdim=True)
        hier[str(res_)] = float(count_components(mk, 2 * res_).mean().item())
    out["structures_by_scale"] = hier
    # varredura de fase: ganho de interação × difusão
    gains, diffs = [0.5, 1.0, 1.5], [0.0, 1.0, 2.0]
    variants = []
    for gg in gains:
        for dd in diffs:
            variants.append(self._variant(genome, lambda p, gg=gg, dd=dd: (p.__setitem__("W", p["W"] * gg),
                                                                          p.__setitem__("diff", p["diff"] * dd)), f"ph{gg}_{dd}"))
    vecs, via = self._desc_runs(variants, self.envs_for(split), derive_seed(seed, "phase"))
    grid = vecs[:, DESCRIPTOR_NAMES.index("persistence")].reshape(len(gains), len(diffs))
    act = vecs[:, DESCRIPTOR_NAMES.index("rel_activity")].reshape(len(gains), len(diffs))
    jumps = max(float(np.abs(np.diff(grid, axis=0)).max(initial=0)), float(np.abs(np.diff(grid, axis=1)).max(initial=0)))
    out["phase_diagram"] = {"gains": gains, "diffusion_scales": diffs, "persistence": grid.round(3).tolist(),
                            "activity": act.round(3).tolist(), "viability": via, "max_jump": jumps,
                            "abrupt_transition_suspected": bool(jumps > 0.4)}
    # histerese ambiental (recurso sobe e desce)
    gb1, st1, _ = self._setup(genome, 1, split, derive_seed(seed, "hyst"))
    st1 = self.sim.run(gb1, st1, self.ps.warmup)
    levels = [0.4, 0.7, 1.0, 1.3, 1.6]
    up, down = [], []
    for seq, store in [(levels, up), (levels[::-1], down)]:
        for lv in seq:
            st1.R0 = torch.full_like(st1.R0, lv)
            U0 = st1.U.clone()
            st1 = self.sim.run(gb1, st1, 32)
            store.append(float((st1.U - U0).abs().mean().item()))
    out["hysteresis"] = {"levels": levels, "activity_up": up, "activity_down": down[::-1],
                         "area": float(np.abs(np.asarray(up) - np.asarray(down[::-1])).sum())}
    # mapa de importância causal: dano em blocos 4x4 e divergência resultante (pares com ruído idêntico)
    nb = 4
    bs = S // nb
    gbm, stm, _ = self._setup(genome, 1, split, derive_seed(seed, "imp"), repeat=nb * nb + 1)
    stm = self.sim.run(gbm, stm, self.ps.warmup)
    mask = torch.zeros(nb * nb + 1, 1, S, S, dtype=torch.bool, device=self.sim.device)
    for k in range(nb * nb):
        i, j = divmod(k, nb)
        mask[k + 1, 0, i * bs:(i + 1) * bs, j * bs:(j + 1) * bs] = True
    stm.damage_cells(mask)
    stm = self.sim.run(gbm, stm, 64)
    ref = stm.U[0]
    imp = ((stm.U[1:] - ref).abs().mean((1, 2, 3)) / (ref.abs().mean() + 1e-9)).cpu().numpy().reshape(nb, nb)
    out["causal_importance_map"] = imp.round(4).tolist()
    out["importance_concentration"] = float(imp.max() / (imp.mean() + 1e-9))
    # localização da memória: decodificação local vs distante vs global
    pats = [self._pattern(np.random.default_rng(seed), self.cfg.channels_max, self.ps.stim_radius + 1) for _ in range(2)]
    c = S // 2
    pool = self.envs_for(split)
    units = []
    k_ = 2 * self.ps.trials
    for i in range(k_):
        units.append({"env": pool[(i // 2) % len(pool)], "seed": derive_seed(seed, "memloc", i), "group": i,
                      "stims": [(h, pats[i % 2], c, c, self.ps.stim_amp * 0.5) for h in range(self.ps.memory_history_steps)]})
    tmem = self.ps.memory_history_steps + self.th.memory_min_delay
    far = (c + S // 4) % S
    feats, _ = self._run_protocol(genome, units, [
        {"name": "local", "kind": "region", "cy": c, "cx": c, "r": self.ps.stim_radius + 2, "times": [tmem]},
        {"name": "far", "kind": "region", "cy": far, "cx": far, "r": self.ps.stim_radius + 2, "times": [tmem]},
        {"name": "global", "kind": "global", "cy": 0, "cx": 0, "r": 0, "times": [tmem]}], tmem, derive_seed(seed, "ml"))
    y = np.arange(k_) % 2
    loc = {nm: decode_report(feats[nm][tmem], y, 2, self.ps.permutations, self.ps.folds, seed)["linear_acc"] for nm in feats}
    out["memory_location_acc"] = loc
    out["memory_distribution"] = ("localized" if loc["local"] - loc["far"] > 0.2 else
                                  ("distributed" if loc["far"] > 0.7 else "weak/none"))
    return TestResult(probe="advanced", candidate_id=genome.genome_id, status=STATUS_PASS, score=float("nan"),
                      details=out, protocol={**prot, "steps": steps}, split=split)


ValidationEngine.probe_advanced = probe_advanced
