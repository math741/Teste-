# %% [markdown]
# ## 5. Observação: descritores comportamentais, triagem populacional e decodificadores externos
# A triagem (fase rápida) roda cada genoma em vários ambientes de **evolução**, cada um com uma cópia contrafactual
# (`U_A(0)=U_B(0)`, perturbação local apenas em B, **mesmo ruído** nas duas cópias) e mede, sem olhar médias apenas:
# atividade, variância, autocorrelação temporal e espacial, espectros, periodicidade, entropia (proxy), compressibilidade,
# número aproximado de estruturas (componentes conexas por propagação de rótulos), persistência de identidade e
# mobilidade (correlação cruzada normalizada com deslocamento), propagação causal e divergência (proxy de Lyapunov).
#
# Decodificadores são **instrumentos científicos externos**: ridge linear (primário, conservador) e kNN (não linear),
# sempre com validação cruzada estratificada, teste de permutação, acurácia balanceada e informação mútua com correção de viés.

# %%
DESCRIPTOR_NAMES = ["rel_activity", "variance", "tac1", "sac", "spatial_spectrum", "dominant_freq", "periodicity",
                    "mobility", "structures", "persistence", "causal_spread", "lyapunov", "entropy",
                    "compressibility", "saturation", "magnitude", "temporal_complexity", "spatial_complexity"]


def count_components(mask: torch.Tensor, max_iters: int) -> torch.Tensor:
    """Componentes conexas aproximadas (vizinhança 8) por propagação de rótulo máximo. mask [N,m,m] bool."""
    N, m, _ = mask.shape
    ids = torch.arange(1, m * m + 1, device=mask.device, dtype=torch.float32).view(1, m, m)
    mf = mask.float()
    lab = ids * mf
    for i in range(max_iters):
        new = F.max_pool2d(lab.unsqueeze(1), 3, 1, 1).squeeze(1) * mf
        if i % 8 == 7 and torch.equal(new, lab):
            lab = new
            break
        lab = new
    roots = (lab == ids) & mask
    return roots.flatten(1).sum(1).float()


def compute_descriptors(Fr: torch.Tensor, Dm: Optional[torch.Tensor], D0: Optional[torch.Tensor], chm: torch.Tensor,
                        sat: torch.Tensor, S: int, rec_every: int, steps_after: int) -> Dict[str, np.ndarray]:
    """Fr: [T,B,C,m,m] quadros registrados; Dm: [T,B,m,m] |U_A-U_B| (contrafactual); chm,sat: [B,C]."""
    T, B, C, m, _ = Fr.shape
    eps = 1e-8
    dev = Fr.device
    nch = chm.sum(1).clamp(min=1)
    absmean = Fr.abs().sum(2).mean((0, 2, 3)) / nch
    dF = (Fr[1:] - Fr[:-1]).abs().sum(2).mean((2, 3)) / nch
    activity = dF.mean(0)
    rel_act = activity / (absmean + 1e-6)
    mag = Fr.pow(2).sum(2).sqrt()
    variance = mag.var(dim=(2, 3)).mean(0)
    X = Fr - Fr.mean(0, keepdim=True)

    def tac(lag):
        if T <= lag:
            return torch.ones(B, device=dev)
        num = (X[lag:] * X[:-lag]).sum((0, 2, 3, 4))
        den = torch.sqrt((X[lag:] ** 2).sum((0, 2, 3, 4)) * (X[:-lag] ** 2).sum((0, 2, 3, 4)))
        return torch.where(den > 1e-10, num / (den + eps), torch.ones_like(num))
    tac1 = tac(1)
    M = mag - mag.mean((2, 3), keepdim=True)
    den = (M * M).sum((0, 2, 3))
    sac = ((M * M.roll(1, 2)).sum((0, 2, 3)) + (M * M.roll(1, 3)).sum((0, 2, 3))) / (2 * den + eps)
    sac = torch.where(den > 1e-10, sac, torch.zeros_like(sac))
    Pw = (torch.fft.fft2(M).abs() ** 2).mean(0)
    fy = torch.fft.fftfreq(m, device=dev).abs()
    kr = torch.sqrt(fy.view(m, 1) ** 2 + fy.view(1, m) ** 2); kr = kr / kr.max()
    spec_c = (Pw * kr).sum((1, 2)) / (Pw.sum((1, 2)) + eps)
    spec_c = torch.where(Pw.sum((1, 2)) > 1e-10, spec_c, torch.zeros_like(spec_c))
    Mt = mag - mag.mean(0, keepdim=True)
    Pt = (torch.fft.rfft(Mt, dim=0).abs() ** 2)[1:]
    Fq = max(2, Pt.shape[0])
    Ptot = Pt.sum((2, 3))
    tot = Ptot.sum(0)
    has_t = tot > 1e-10
    dom = (Ptot.argmax(0).float() + 1) / Fq
    periodicity = torch.where(has_t, Ptot.max(0).values / (tot + eps), torch.zeros_like(tot))
    pc = Pt / (Pt.sum(0, keepdim=True) + eps)
    ent_c = -(pc * torch.log(pc + 1e-12)).sum(0) / math.log(Fq)
    wcell = Pt.sum(0)
    tcell = (ent_c * wcell).sum((1, 2)) / (wcell.sum((1, 2)) + eps)
    temporal_complexity = torch.where(has_t, tcell * tac1.clamp(0, 1), torch.zeros_like(tcell))
    # entropia (proxy) do histograma de magnitude
    nb = 16
    smag = torch.sqrt((sat ** 2 * chm).sum(1)).clamp(min=1e-3)
    q = (mag / smag.view(1, B, 1, 1)).clamp(0, 0.9999) * nb
    idx = q.long().permute(1, 0, 2, 3).reshape(B, -1)
    hist = torch.zeros(B, nb, device=dev).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    pp = hist / hist.sum(1, keepdim=True)
    entropy = -(pp * torch.log(pp + 1e-12)).sum(1) / math.log(nb)
    spatial_complexity = entropy * sac.clamp(0, 1)
    satm = (Fr.abs() > 0.95 * sat.view(1, B, C, 1, 1)) & (chm.view(1, B, C, 1, 1) > 0)
    saturation = satm.float().sum((0, 2, 3, 4)) / (T * nch * m * m)
    # persistência de identidade + mobilidade (NCC com deslocamento)
    lag = max(1, T // 4)
    A_, B_ = M[:-lag], M[lag:]
    xc = torch.fft.ifft2(torch.conj(torch.fft.fft2(A_)) * torch.fft.fft2(B_)).real
    nA = torch.sqrt((A_ * A_).sum((2, 3))); nB = torch.sqrt((B_ * B_).sum((2, 3)))
    ncc = xc / (nA * nB + eps).unsqueeze(-1).unsqueeze(-1)
    peak, arg = ncc.flatten(2).max(2)
    valid = (nA > 1e-5) & (nB > 1e-5)
    peak = torch.where(valid, peak, torch.zeros_like(peak))
    dy = (arg // m).float(); dx = (arg % m).float()
    dy = torch.where(dy > m // 2, dy - m, dy); dx = torch.where(dx > m // 2, dx - m, dx)
    disp = torch.sqrt(dy ** 2 + dx ** 2) * (S / m) / (lag * rec_every)
    moving = (peak > 0.5) & (disp > 0)
    vel = (disp * moving).sum(0) / moving.sum(0).clamp(min=1)
    dirc = torch.sqrt((dy * moving).sum(0) ** 2 + (dx * moving).sum(0) ** 2) / (
        (torch.sqrt(dy ** 2 + dx ** 2) * moving).sum(0) + eps)
    mobility = (vel / 0.25).clamp(0, 1) * moving.float().mean(0)
    persistence = peak.mean(0).clamp(0, 1) * (variance > 1e-7).float()
    # estruturas
    last = mag[-2:] if T >= 2 else mag[-1:]
    mm = last.mean((2, 3), keepdim=True); ss = last.std((2, 3), keepdim=True)
    mask = (last > mm + ss) & (last > 0.05 * smag.view(1, B, 1, 1))
    nc = count_components(mask.reshape(-1, m, m), 2 * m).view(-1, B).mean(0)
    structures = nc
    # causalidade
    if Dm is not None:
        Dend = Dm[-1]
        thr = 1e-3 + 1e-2 * absmean.view(B, 1, 1)
        spread = (Dend > thr).float().mean((1, 2))
        Dmean_end = Dm[-1].mean((1, 2))
        lyap = (torch.log(Dmean_end + 1e-12) - torch.log(D0 + 1e-12)) / max(1, steps_after)
        c = (m - 1) / 2.0
        yy, xx = torch.meshgrid(torch.arange(m, device=dev), torch.arange(m, device=dev), indexing="ij")
        dist = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2).float() * (S / m)
        rmax = (dist.unsqueeze(0) * (Dend > thr).float()).amax((1, 2))
        speed = rmax / max(1, steps_after)
    else:
        spread = lyap = speed = torch.zeros(B, device=dev)
    # compressibilidade (zlib) — CPU
    qm = (mag[-4:] / (mag[-4:].amax((0, 2, 3), keepdim=True) + 1e-9) * 255).clamp(0, 255).to(torch.uint8)
    qm = qm.permute(1, 0, 2, 3).contiguous().cpu().numpy()
    comp = np.array([len(zlib.compress(qm[b].tobytes(), 6)) / max(1, qm[b].size) for b in range(B)], np.float32)
    out = {
        "rel_activity": rel_act, "activity": activity, "variance": variance, "tac1": tac1, "sac": sac,
        "spatial_spectrum": spec_c, "dominant_freq": dom, "periodicity": periodicity, "mobility": mobility,
        "velocity": vel, "direction_consistency": dirc, "shape_retention": peak.mean(0),
        "structures": structures, "persistence": persistence, "causal_spread": spread, "lyapunov": lyap,
        "spread_speed": speed, "entropy": entropy, "saturation": saturation, "magnitude": absmean,
        "temporal_complexity": temporal_complexity, "spatial_complexity": spatial_complexity,
    }
    res = {k: torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy() for k, v in out.items()}
    res["compressibility"] = comp
    return res


def descriptor_vector(d: Dict[str, float]) -> np.ndarray:
    """Vetor comportamental normalizado ~[0,1] para novelty / MAP / surpresa."""
    sq = lambda x: x / (1.0 + abs(x))
    v = [sq(d["rel_activity"] * 20), math.tanh(d["variance"] * 10), (d["tac1"] + 1) / 2, (d["sac"] + 1) / 2,
         d["spatial_spectrum"], d["dominant_freq"], d["periodicity"], d["mobility"],
         sq(d["structures"] / 4.0), d["persistence"], d["causal_spread"], 1 / (1 + math.exp(-d["lyapunov"] * 30)),
         d["entropy"], min(1.0, d["compressibility"]), d["saturation"], math.tanh(d["magnitude"]),
         d["temporal_complexity"], d["spatial_complexity"]]
    return np.nan_to_num(np.asarray(v, np.float32), nan=0.0)


def classify_viability(d: Dict[str, float], th: Thresholds, bad: bool) -> str:
    if bad:
        return "NUMERICALLY_UNSTABLE"
    if d["saturation"] > th.explosive_saturation:
        return "EXPLOSIVE"
    if d["magnitude"] < th.dead_abs:
        return "DEAD"
    if d["rel_activity"] < th.frozen_rel_activity:
        return "FROZEN"
    if d["tac1"] < th.noise_tac and d["sac"] < th.noise_sac and d["entropy"] > th.noise_entropy:
        return "PURE_NOISE"
    return "VIABLE"


def quality_objectives(d: Dict[str, float], th: Thresholds, probe_bonus: float = 0.0) -> np.ndarray:
    """Várias dimensões independentes (nunca uma única métrica)."""
    frozen = d["rel_activity"] < th.frozen_rel_activity * 10
    chaos = 1.0 / (1.0 + math.exp((d["lyapunov"] - th.chaos_lyap) / 0.01))
    return np.asarray([
        d["persistence"] * (0.5 if frozen else 1.0),
        d["temporal_complexity"],
        d["spatial_complexity"],
        d["causal_spread"] * chaos,
        d["mobility"],
        min(1.0, d["structures"] / 4.0),
        probe_bonus,
    ], np.float32)


QUALITY_WEIGHTS = np.asarray([0.2, 0.2, 0.15, 0.2, 0.1, 0.15, 0.5], np.float32)


class Screener:
    """Triagem populacional rápida, em batch, multi-ambiente, com par contrafactual."""

    def __init__(self, sim: Simulator, cfg: GenesisConfig):
        self.sim, self.cfg = sim, cfg

    def screen(self, genomes: List[Genome], envs: List[Environment], seed_base: int,
               steps: Optional[int] = None, S: Optional[int] = None, n_envs: Optional[int] = None,
               hook: Optional[Callable] = None, env_choice_seed: Optional[int] = None,
               label: Optional[str] = None) -> List[Dict[str, Any]]:
        cfg = self.cfg
        steps = steps or cfg.screen_steps
        S = S or cfg.world_size
        n_envs = min(n_envs or cfg.screen_envs, len(envs))
        rng = np.random.default_rng(derive_seed(seed_base, "envchoice") if env_choice_seed is None else env_choice_seed)
        jobs = []
        for gi, g in enumerate(genomes):
            chosen = rng.choice(len(envs), size=n_envs, replace=False)
            for ei in chosen:
                jobs.append((gi, envs[int(ei)], derive_seed(seed_base, g.genome_id, envs[int(ei)].env_id)))
        per = max(1, cfg.batch_size // 2)
        rows: List[Dict[str, Any]] = []
        n_chunks = max(1, int(math.ceil(len(jobs) / per)))
        for ci, c0 in enumerate(range(0, len(jobs), per)):
            base = None
            if label is not None:
                base = ci * steps
                live_progress(phase=label, total=n_chunks * steps, done=base,
                              detail=f"lote {ci + 1}/{n_chunks} · {len(genomes)} genomas × {n_envs} ambientes "
                                     f"(+ cópia contrafactual de cada universo)")
            rows.extend(self._screen_chunk(genomes, jobs[c0:c0 + per], steps, S, seed_base + c0, hook, base))
        # agrega por genoma (média entre ambientes; instável se qualquer ambiente instável)
        agg: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            agg.setdefault(r["gi"], []).append(r)
        out = []
        for gi in range(len(genomes)):
            rs = agg.get(gi, [])
            d = {k: float(np.mean([r[k] for r in rs])) for k in rs[0].keys() if k not in ("gi", "env", "seed", "bad")}
            d["bad"] = any(r["bad"] for r in rs)
            d["envs"] = [r["env"] for r in rs]
            d["seeds"] = [r["seed"] for r in rs]
            d["viability"] = classify_viability(d, cfg.thresholds, d["bad"])
            d["env_viability"] = [classify_viability(r, cfg.thresholds, r["bad"]) for r in rs]
            out.append(d)
        return out

    def _screen_chunk(self, genomes, jobs, steps, S, run_seed, hook, progress_base: Optional[int] = None):
        cfg, sim = self.cfg, self.sim
        n = len(jobs)
        gl = [genomes[gi] for gi, _, _ in jobs]
        gb = sim.prepare(gl + gl, S)
        envs = [e for _, e, _ in jobs]
        seeds = [s for _, _, s in jobs]
        st = sim.init_state(gb, envs + envs, seeds + seeds, groups=list(range(n)) * 2, run_seed=run_seed)
        burn = min(cfg.screen_burn_in, steps // 2)
        for t in range(burn):
            st = sim.step(gb, st)
            if hook is not None:
                hook(st)
            if progress_base is not None:
                live_progress(done=progress_base + t + 1)
        # perturbação contrafactual local somente na cópia B
        r = 2
        c = S // 2
        delta = torch.zeros_like(st.U[n:])
        delta[:, :, c - r:c + r + 1, c - r:c + r + 1] = 0.5
        st.U[n:] = st.U[n:] + delta * gb.chm[n:]
        m = min(cfg.metric_res, S)
        pool = lambda X: F.adaptive_avg_pool2d(X, m) if m != S else X
        D0 = pool((st.U[:n] - st.U[n:]).abs().sum(1, keepdim=True)).squeeze(1).mean((1, 2))
        frames, dms = [], []
        rest = steps - burn
        for t in range(rest):
            st = sim.step(gb, st)
            if hook is not None:
                hook(st)
            if progress_base is not None:
                live_progress(done=progress_base + burn + t + 1)
            if t % cfg.record_every == cfg.record_every - 1 or t == rest - 1:
                frames.append(pool(st.U[:n]).float())
                dms.append(pool((st.U[:n] - st.U[n:]).abs().sum(1, keepdim=True)).squeeze(1))
        Fr = torch.stack(frames); Dm = torch.stack(dms)
        del frames, dms
        chm = gb.chm[:n].view(n, -1).clone(); sat = gb.sat[:n].view(n, -1).clone()
        bad = (st.bad[:n] | st.bad[n:]).cpu().numpy()
        del st, gb                                   # libera o estado do batch antes da análise
        parts = []
        for a in range(0, n, 256):                   # descritores em sub-lotes: memória limitada mesmo com batch grande
            b = min(n, a + 256)
            parts.append(compute_descriptors(Fr[:, a:b], Dm[:, a:b], D0[a:b], chm[a:b], sat[a:b], S, cfg.record_every, rest))
        desc = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        out = []
        for i in range(n):
            row = {k: float(v[i]) for k, v in desc.items()}
            row.update({"gi": jobs[i][0], "env": jobs[i][1].env_id, "seed": jobs[i][2], "bad": bool(bad[i])})
            out.append(row)
        return out


# ----------------------------------------------------------------------------
# Decodificadores externos (instrumentos)
# ----------------------------------------------------------------------------
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_predict(Xtr, ytr, Xte, n_classes, lam=1.0):
    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))]); Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    Y = np.eye(n_classes)[ytr]
    n, d = Xtr.shape
    if n < d:   # forma dual
        Kt = Xtr @ Xtr.T
        alpha = np.linalg.solve(Kt + lam * np.eye(n), Y)
        scores = Xte @ (Xtr.T @ alpha)
    else:
        Wt = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Y)
        scores = Xte @ Wt
    return scores.argmax(1)


def knn_predict(Xtr, ytr, Xte, n_classes, k=5):
    Xtr, Xte = _standardize(Xtr, Xte)
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    k = max(1, min(k, Xtr.shape[0]))
    nn = np.argsort(d, axis=1)[:, :k]
    votes = np.zeros((Xte.shape[0], n_classes))
    for j in range(k):
        votes[np.arange(Xte.shape[0]), ytr[nn[:, j]]] += 1.0 / (1 + j * 0.01)
    return votes.argmax(1)


def stratified_folds(y, k, rng):
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def balanced_accuracy(y, yp, n_classes):
    accs = [np.mean(yp[y == c] == c) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(accs)) if accs else 0.0


def cv_predictions(X, y, n_classes, decoder, k, seed):
    rng = np.random.default_rng(seed)
    k = max(2, min(k, int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 2))
    folds = stratified_folds(y, k, rng)
    yp = np.zeros_like(y)
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        yp[te] = decoder(X[tr], y[tr], X[te], n_classes)
    return yp


def mutual_info_bits(y, yp, n_classes) -> float:
    """MI plug-in da matriz de confusão com correção de Miller–Madow (viés reconhecido)."""
    conf = np.zeros((n_classes, n_classes))
    for a, b in zip(y, yp):
        conf[a, b] += 1
    n = conf.sum()
    if n == 0:
        return 0.0
    pj = conf / n
    px = pj.sum(1, keepdims=True); py = pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (px @ py)[nz])).sum())
    k_joint, k_x, k_y = nz.sum(), (px > 0).sum(), (py > 0).sum()
    mm = (k_joint - k_x - k_y + 1) / (2 * n * math.log(2))
    return max(0.0, mi - mm)


def decode_report(X: np.ndarray, y: np.ndarray, n_classes: int, n_perm: int, folds: int, seed: int,
                  nonlinear: bool = True) -> Dict[str, Any]:
    """Acurácia balanceada CV (linear e kNN), p-valor por permutação, confusão, MI."""
    X = np.nan_to_num(np.asarray(X, np.float64)); y = np.asarray(y, int)
    chance = 1.0 / n_classes
    out: Dict[str, Any] = {"n": int(len(y)), "chance": chance, "n_classes": n_classes,
                           "method": "balanced acc, stratified CV (ridge linear + kNN), label-permutation null (empirical + Gaussian tail)"}
    if len(np.unique(y)) < 2 or X.shape[0] < 2 * n_classes:
        out.update({"linear_acc": chance, "linear_p": 1.0, "nonlinear_acc": chance, "nonlinear_p": 1.0, "mi_bits": 0.0})
        return out
    if np.all(X.std(0) < 1e-9):
        out.update({"linear_acc": chance, "linear_p": 1.0, "nonlinear_acc": chance, "nonlinear_p": 1.0, "mi_bits": 0.0,
                    "note": "features constantes"})
        return out
    rng = np.random.default_rng(seed)
    for name, dec in [("linear", ridge_predict)] + ([("nonlinear", knn_predict)] if nonlinear else []):
        yp = cv_predictions(X, y, n_classes, dec, folds, seed)
        acc = balanced_accuracy(y, yp, n_classes)
        null = []
        for i in range(n_perm):
            if i % 10 == 0:
                live_progress()
            yperm = rng.permutation(y)
            ypp = cv_predictions(X, yperm, n_classes, dec, folds, seed + 1 + i)
            null.append(balanced_accuracy(yperm, ypp, n_classes))
        null = np.asarray(null)
        p_emp = (np.sum(null >= acc - 1e-12) + 1) / (n_perm + 1)
        # cauda gaussiana da distribuição nula de permutação: permite p pequenos com poucas permutações;
        # só é usada quando a acurácia observada supera TODAS as permutações (senão vale o p empírico).
        sd = max(float(null.std()), 1e-3) if null.size else 1.0
        z = (acc - (float(null.mean()) if null.size else chance)) / sd
        p_par = 0.5 * math.erfc(z / math.sqrt(2))
        p = p_par if (null.size and acc > null.max()) else max(p_emp, p_par)
        out[f"{name}_acc"] = acc
        out[f"{name}_p"] = float(p)
        out[f"{name}_p_empirical"] = float(p_emp)
        out[f"{name}_null_mean"] = float(null.mean()) if null.size else chance
        if name == "linear":
            out["mi_bits"] = mutual_info_bits(y, yp, n_classes)
            conf = np.zeros((n_classes, n_classes), int)
            for a, b in zip(y, yp):
                conf[a, b] += 1
            out["confusion"] = conf.tolist()
            out["effect_size"] = float((acc - out[f"{name}_null_mean"]) / (null.std() + 1e-6)) if null.size else 0.0
    if not nonlinear:
        out["nonlinear_acc"], out["nonlinear_p"] = float("nan"), 1.0
    return out


def norm_acc(acc: float, chance: float) -> float:
    return float((acc - chance) / (1 - chance))
