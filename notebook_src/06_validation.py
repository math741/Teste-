# %% [markdown]
# ## 6. Validation Engine (congelado e versionado) — biblioteca de probes
# O universo **não controla seu avaliador**. Este módulo é separado do motor evolutivo, tem versão própria
# (`VE-1.0`), cada probe tem versão, cada resultado guarda o protocolo completo (sementes, ambientes, tempos, tamanho)
# para ser reproduzido por `reproduce_discovery`. Resultados ausentes são `NOT_TESTED` (zero significa "testado e falhou").
# Estados possíveis: `NOT_TESTED, RUNNING, PASS, FAIL, INCONCLUSIVE, ERROR`. Inconclusivo é um resultado válido.
#
# | probe | pergunta | controles |
# |---|---|---|
# | causality | perturbação local produz efeito reproduzível e seletivo (não só caos)? | cópia contrafactual com ruído idêntico, repetibilidade entre sementes, seletividade entre 2 estímulos |
# | memory | o estado futuro permite inferir qual história ocorreu, após T passos sem estímulo? | grupo nulo (sem estímulo), permutação de rótulos, Holm entre atrasos, controle +/− no boot |
# | channel | símbolos aplicados em A são decodificáveis em B (distância d)? | mapeamento símbolo→estímulo aleatorizado, leitura pré-estímulo (vazamento), estímulo nulo |
# | transformation | a região de saída torna linearmente decodificável uma função NÃO linear (XOR) das entradas? | baseline linear das entradas (XOR impossível), leitura pré-estímulo |
# | composition | A→B e B→C se compõem (paridade de 3 entradas em C, XOR intermediário em M)? | idem |
# | generalization | decodificador treinado em condições vistas funciona em amplitudes, posições, sementes e ambientes NÃO vistos? | split evolução/validação; t* escolhido só no treino |
# | recovery | estrutura persistente danificada (10–90%) volta em geometria, estado interno e trajetória? | ramo de referência sem dano com ruído idêntico; recuperação relativa ao dano imediato |
# | robustness | comportamento sobrevive a ruído, sal-e-pimenta, deleção, drift de parâmetro, dt, borda, estado inicial, tamanho? | baseline com mesmas sementes |
# | adaptation | o sistema muda comportamento quando o ambiente muda e mantém função? | condição sem mudança (deriva natural) |
# | replication / ablation | o resultado se repete com sementes independentes? o que realmente importa? | genoma congelado; zero evolução durante validação |

# %%
@dataclass
class TestResult:
    probe: str
    candidate_id: str
    status: str
    score: float = float("nan")
    p_value: float = float("nan")
    effect_size: float = float("nan")
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    details: Dict[str, Any] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)
    version: str = ""
    result_id: str = ""
    timestamp: str = ""
    duration_s: float = 0.0
    split: str = "evolution"
    thresholds_hash: str = ""

    def __post_init__(self):
        if not self.result_id:
            self.result_id = new_id("R")
        if not self.timestamp:
            self.timestamp = iso_now()

    def to_dict(self):
        return to_jsonable(asdict(self))

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["ci"] = tuple(x if x is not None else float("nan") for x in d.get("ci", (None, None)))
        for k in ["score", "p_value", "effect_size"]:
            if d.get(k) is None:
                d[k] = float("nan")
        return TestResult(**{k: v for k, v in d.items() if k in TestResult.__dataclass_fields__})


class ValidationEngine:
    VERSION = VALIDATION_ENGINE_VERSION
    PROBE_VERSIONS = {"causality": "1.0", "memory": "1.0", "channel": "1.0", "transformation": "1.0",
                      "composition": "1.0", "generalization": "1.0", "recovery": "1.0", "robustness": "1.0",
                      "adaptation": "1.0", "replication": "1.0", "ablation": "1.0", "long_screen": "1.0",
                      "advanced": "1.0"}

    def __init__(self, sim: Simulator, cfg: GenesisConfig, envs: List[Environment], screener: Screener,
                 space: ParamSpace, curriculum: Optional[Dict[str, float]] = None):
        self.sim, self.cfg, self.envs, self.screener, self.space = sim, cfg, envs, screener, space
        self.curriculum = curriculum if curriculum is not None else {}
        self.th = cfg.thresholds
        self.ps = cfg.probe
        self._frozen_hash = cfg.thresholds_hash()

    # ------------------------------------------------------------------
    def envs_for(self, split: str) -> List[Environment]:
        e = [x for x in self.envs if x.split == split]
        return e if e else [x for x in self.envs if x.split == "evolution"]

    def channel_distance(self, S: int) -> int:
        d = int(self.ps.channel_distance + self.curriculum.get("channel_distance_add", 0))
        return int(max(2 * self.ps.stim_radius + 2, min(d, S // 2 - self.ps.obs_radius - 1)))

    def memory_delays(self) -> List[int]:
        mult = self.curriculum.get("memory_delay_mult", 1.0)
        return sorted(set(max(1, int(round(d * mult))) if d > 1 else d for d in self.ps.memory_delays))

    def run_probe(self, name: str, genome: Genome, seed: int, **kw) -> TestResult:
        """Wrapper: tempo, protocolo completo, tratamento de erro (sem falhas silenciosas)."""
        t0 = time.time()
        fn = getattr(self, f"probe_{name}")
        protocol = {"probe": name, "probe_version": self.PROBE_VERSIONS.get(name, "?"), "engine": self.VERSION,
                    "seed": int(seed), "kwargs": to_jsonable(kw), "world_size": kw.get("S", self.cfg.world_size),
                    "curriculum": dict(self.curriculum), "genome_hash": genome.hash(),
                    "device": self.sim.device.type}
        try:
            res: TestResult = fn(genome, seed, **kw)
        except Exception as e:
            if is_oom_error(e):
                raise
            res = TestResult(probe=name, candidate_id=genome.genome_id, status=STATUS_ERROR,
                             details={"error": repr(e), "traceback": traceback.format_exc()[-3000:]})
        res.probe = name
        res.candidate_id = genome.genome_id
        res.version = f"{self.VERSION}/{name}-{self.PROBE_VERSIONS.get(name, '?')}"
        res.protocol = {**protocol, **res.protocol}
        res.duration_s = time.time() - t0
        res.thresholds_hash = self._frozen_hash
        res.split = kw.get("split", res.split)
        return res

    # ------------------------------------------------------------------
    def _setup(self, genome: Genome, n_unique: int, split: str, seed: int, repeat: int = 1, S: Optional[int] = None,
               envs: Optional[List[Environment]] = None, genomes: Optional[List[Genome]] = None):
        S = S or self.cfg.world_size
        pool = envs if envs is not None else self.envs_for(split)
        env_u = [pool[i % len(pool)] for i in range(n_unique)]
        seeds_u = [derive_seed(seed, "trial", i) for i in range(n_unique)]
        gl = genomes if genomes is not None else [genome] * (n_unique * repeat)
        gb = self.sim.prepare(gl, S)
        st = self.sim.init_state(gb, env_u * repeat, seeds_u * repeat, groups=list(range(n_unique)) * repeat,
                                 run_seed=derive_seed(seed, "run"))
        return gb, st, {"S": S, "envs": [e.env_id for e in env_u], "seeds": seeds_u, "split": split}

    @staticmethod
    def _pattern(rng: np.random.Generator, C: int, r: int) -> np.ndarray:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = ((yy ** 2 + xx ** 2) <= r * r + 0.5).astype(np.float32)
        p = rng.choice([-1.0, 1.0], size=(C, 2 * r + 1, 2 * r + 1)).astype(np.float32)
        return p * disk[None]

    def _stim_field(self, B: int, C: int, S: int, items: List[Tuple[int, np.ndarray, int, int, float]]) -> np.ndarray:
        fld = np.zeros((B, C, S, S), np.float32)
        for (b, pat, cy, cx, amp) in items:
            r = pat.shape[-1] // 2
            ys = (np.arange(-r, r + 1) + cy) % S; xs = (np.arange(-r, r + 1) + cx) % S
            fld[b][:, ys[:, None], xs[None, :]] += amp * pat
        return fld

    def _region_feats(self, U: torch.Tensor, cy: int, cx: int, r: int, pool: int = 3) -> np.ndarray:
        S = U.shape[-1]
        ys = torch.as_tensor((np.arange(-r, r + 1) + cy) % S, device=U.device)
        xs = torch.as_tensor((np.arange(-r, r + 1) + cx) % S, device=U.device)
        crop = U.index_select(2, ys).index_select(3, xs)
        f = F.adaptive_avg_pool2d(crop, min(pool, 2 * r + 1))
        return f.flatten(1).float().cpu().numpy()

    def _global_feats(self, U: torch.Tensor, pool: int = 6) -> np.ndarray:
        return F.adaptive_avg_pool2d(U, min(pool, U.shape[-1])).flatten(1).float().cpu().numpy()

    def _run_protocol(self, genome: Genome, units: List[Dict[str, Any]], readouts: List[Dict[str, Any]], T_total: int,
                      seed: int, S: Optional[int] = None, noise_level: float = 0.0,
                      env_change: Optional[Dict[str, Any]] = None, damage: Optional[Dict[str, Any]] = None,
                      warmup: Optional[int] = None):
        """Motor genérico de protocolo estímulo→leitura.
        units: [{env, seed, group, stims:[(t, pattern, cy, cx, amp)]}] ; readouts: [{name, kind, cy, cx, r, times}]."""
        S = S or self.cfg.world_size
        B = len(units)
        gb = self.sim.prepare([genome] * B, S)
        st = self.sim.init_state(gb, [u["env"] for u in units], [u["seed"] for u in units],
                                 groups=[u["group"] for u in units], run_seed=derive_seed(seed, "protocol-run"))
        C = gb.C
        st = self.sim.run(gb, st, self.ps.warmup if warmup is None else warmup)
        if env_change is not None and env_change.get("t", 0) < 0:
            self._apply_env_change(st, env_change, units)
        if damage is not None:
            rng = np.random.default_rng(derive_seed(seed, "damage"))
            mask = np.zeros((B, 1, S, S), bool)
            y0, y1, x0, x1 = damage["box"]
            sub = rng.random((B, 1, y1 - y0, x1 - x0)) < damage["frac"]
            mask[:, :, y0:y1, x0:x1] = sub
            st.damage_cells(torch.as_tensor(mask, device=self.sim.device))
            st = self.sim.run(gb, st, damage.get("recover", self.ps.recovery_steps))
        if noise_level > 0:
            st.ext_noise = st.ext_noise + noise_level
        sched: Dict[int, List] = {}
        for b, u in enumerate(units):
            for (t, pat, cy, cx, amp) in u.get("stims", []):
                sched.setdefault(int(t), []).append((b, pat, cy, cx, amp))
        feats: Dict[str, Dict[int, np.ndarray]] = {ro["name"]: {} for ro in readouts}
        times_needed = sorted(set(t for ro in readouts for t in ro["times"]))
        last_t = max([T_total] + times_needed)
        act_last = None
        for t in range(last_t + 1):
            if env_change is not None and env_change.get("t", -1) == t:
                self._apply_env_change(st, env_change, units)
            for ro in readouts:
                if t in ro["times"]:
                    if ro["kind"] == "global":
                        feats[ro["name"]][t] = self._global_feats(st.U)
                    else:
                        feats[ro["name"]][t] = self._region_feats(st.U, ro["cy"], ro["cx"], ro["r"], ro.get("pool", 3))
            if t in sched:
                fld = self._stim_field(B, C, S, sched[t])
                st.U = st.U + torch.as_tensor(fld, device=self.sim.device) * gb.chm
            if t == last_t:
                break
            Uold = st.U
            st = self.sim.step(gb, st)
            if t == last_t - 1:
                act_last = (st.U - Uold).abs().mean().item()
        return feats, {"bad": st.bad.cpu().numpy().tolist(), "final_activity": act_last,
                       "magnitude": st.U.abs().mean().item()}

    def _apply_env_change(self, st: SimState, ch: Dict[str, Any], units):
        sel = torch.as_tensor([1.0 if u.get("changed", True) else 0.0 for u in units], device=self.sim.device).view(-1, 1, 1, 1)
        st.R0 = st.R0 * (1 - sel) + st.R0 * ch.get("resource_mult", 0.5) * sel
        st.ext_noise = st.ext_noise + sel * ch.get("noise_add", 0.01)

    def _series(self, feats_t: Dict[int, np.ndarray], idx: np.ndarray, y: np.ndarray, n_classes: int, seed: int,
                times: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        out = []
        for t in (times or sorted(feats_t.keys())):
            rep = decode_report(feats_t[t][idx], y, n_classes, self.ps.permutations, self.ps.folds, derive_seed(seed, t))
            rep["t"] = int(t)
            out.append(rep)
        adj = holm_correction([r["linear_p"] for r in out])
        for r, a in zip(out, adj):
            r["p_holm"] = a
            r["norm_acc"] = norm_acc(r["linear_acc"], r["chance"])
            r["sig"] = bool(a < self.th.alpha and r["norm_acc"] >= self.th.min_norm_acc)
            r["nonlinear_only"] = bool((not r["sig"]) and r.get("nonlinear_p", 1.0) < self.th.alpha / max(1, len(out)))
        return out

    # ------------------------------------------------------------------
    # LEVEL 2 — CAUSALIDADE
    # ------------------------------------------------------------------
    def probe_causality(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                        steps: Optional[int] = None, S: Optional[int] = None) -> TestResult:
        n = trials or self.ps.causal_trials; T = steps or self.ps.causal_steps
        gb, st, prot = self._setup(genome, n, split, seed, repeat=3, S=S)
        S = gb.S; C = gb.C
        rng = np.random.default_rng(derive_seed(seed, "causal-patterns"))
        r = self.ps.stim_radius
        p1, p2 = self._pattern(rng, C, r), self._pattern(rng, C, r)
        st = self.sim.run(gb, st, self.ps.warmup)
        c = S // 2
        items = [(n + i, p1, c, c, self.ps.stim_amp) for i in range(n)] + [(2 * n + i, p2, c, c, self.ps.stim_amp) for i in range(n)]
        st.U = st.U + torch.as_tensor(self._stim_field(gb.B, C, S, items), device=self.sim.device) * gb.chm
        D_start = (st.U[n:2 * n] - st.U[:n]).abs().sum(1).mean().item()
        m = min(self.cfg.metric_res, S)
        maps = []
        snap_every = max(1, T // 8)
        for t in range(T):
            st = self.sim.step(gb, st)
            if t % snap_every == snap_every - 1:
                maps.append(F.adaptive_avg_pool2d((st.U[n:n + 1] - st.U[:1]).abs().sum(1, keepdim=True), m)[0, 0].cpu().numpy())
        U0, U1, U2 = st.U[:n], st.U[n:2 * n], st.U[2 * n:]
        D1 = (U1 - U0).abs().sum(1)
        yy, xx = np.mgrid[0:S, 0:S]
        outside = torch.as_tensor(np.sqrt((yy - c) ** 2 + (xx - c) ** 2) > r + 2, device=self.sim.device)
        thr = 1e-3 + 1e-2 * U0.abs().mean().item()
        spread = ((D1 > thr) & outside).float().sum((1, 2)) / outside.float().sum()
        strength = float(spread.mean().item())
        Dm = F.adaptive_avg_pool2d(D1.unsqueeze(1), m).flatten(1).cpu().numpy()
        Dn = Dm - Dm.mean(1, keepdims=True); Dn = Dn / (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-12)
        cm = Dn @ Dn.T
        rep_corr = float((cm.sum() - np.trace(cm)) / max(1, n * (n - 1)))
        Xs = np.concatenate([self._global_feats(U1 - U0), self._global_feats(U2 - U0)])
        ys = np.array([0] * n + [1] * n)
        sel = decode_report(Xs, ys, 2, self.ps.permutations, self.ps.folds, derive_seed(seed, "sel"))
        D_end = D1.mean().item()
        lyap = (math.log(D_end + 1e-12) - math.log(D_start + 1e-12)) / T
        if not np.isfinite(rep_corr):
            rep_corr = 0.0
        sel_ok = sel["linear_p"] < self.th.alpha and norm_acc(sel["linear_acc"], 0.5) >= self.th.min_norm_acc
        if bool(st.bad.any()):
            status = STATUS_ERROR
        elif strength < self.th.causal_spread_probe:
            status = STATUS_FAIL
        elif rep_corr >= self.th.causal_repeatability or sel_ok:
            status = STATUS_PASS
        else:
            status = STATUS_INCONCLUSIVE
        score = float(min(1.0, strength * 3) * max(rep_corr, norm_acc(sel["linear_acc"], 0.5), 0.0))
        lo, hi = bootstrap_ci(spread.cpu().numpy())
        return TestResult(probe="causality", candidate_id=genome.genome_id, status=status, score=score,
                          p_value=sel["linear_p"], effect_size=sel.get("effect_size", float("nan")), ci=(lo, hi),
                          details={"spread_mean": strength, "spread_per_trial": spread.cpu().numpy().tolist(),
                                   "repeatability": rep_corr, "selectivity_acc": sel["linear_acc"],
                                   "selectivity_p": sel["linear_p"], "lyapunov_like": lyap,
                                   "chaos_suspected": bool(strength >= self.th.causal_spread_probe and rep_corr < 0.2 and not sel_ok),
                                   "causal_map": [mp.astype(np.float16) for mp in maps], "map_times": [snap_every * (i + 1) for i in range(len(maps))]},
                          protocol={**prot, "trials": n, "steps": T}, split=split)

    # ------------------------------------------------------------------
    # LEVEL 3 — MEMÓRIA
    # ------------------------------------------------------------------
    def probe_memory(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                     delays: Optional[List[int]] = None, S: Optional[int] = None) -> TestResult:
        n = trials or self.ps.trials
        delays = sorted(delays or self.memory_delays())
        S = S or self.cfg.world_size
        C = self.cfg.channels_max
        rng = np.random.default_rng(derive_seed(seed, "memory-patterns"))
        r = self.ps.stim_radius + 1
        pats = [self._pattern(rng, C, r), self._pattern(rng, C, r)]
        pool = self.envs_for(split)
        c = S // 2
        H = self.ps.memory_history_steps
        units = []
        labels = []
        for i in range(4 * n):
            lab = i % 2
            stim = i < 2 * n
            stims = [(h, pats[lab], c, c, self.ps.stim_amp * 0.5) for h in range(H)] if stim else []
            units.append({"env": pool[(i // 2) % len(pool)], "seed": derive_seed(seed, "mem", i), "group": i, "stims": stims})
            labels.append(lab)
        times = [H + d for d in delays]
        feats, info = self._run_protocol(genome, units, [{"name": "g", "kind": "region", "cy": c, "cx": c,
                                                          "r": r + 2, "pool": 4, "times": times}],
                                         max(times), seed, S=S)
        y = np.asarray(labels)
        idx_s, idx_n = np.arange(2 * n), np.arange(2 * n, 4 * n)
        ser = self._series(feats["g"], idx_s, y[idx_s], 2, derive_seed(seed, "ms"), times)
        nul = self._series(feats["g"], idx_n, y[idx_n], 2, derive_seed(seed, "mn"), times)
        curve = []
        for d, a, b in zip(delays, ser, nul):
            curve.append({"T": int(d), "acc": a["linear_acc"], "p_holm": a["p_holm"], "sig": a["sig"],
                          "nonlinear_acc": a.get("nonlinear_acc"), "null_acc": b["linear_acc"], "null_sig": b["sig"],
                          "mi_bits": a.get("mi_bits", 0.0)})
        long_sig = [cv for cv in curve if cv["sig"] and cv["T"] >= self.th.memory_min_delay]
        any_sig = [cv for cv in curve if cv["sig"]]
        leak = any(cv["null_sig"] for cv in curve)
        nonlin_only = any(r_["nonlinear_only"] for r_ in ser if r_["t"] - H >= self.th.memory_min_delay)
        if any(info["bad"]):
            status = STATUS_ERROR
        elif leak:
            status = STATUS_INCONCLUSIVE
        elif long_sig:
            status = STATUS_PASS
        elif nonlin_only:
            status = STATUS_INCONCLUSIVE
        else:
            status = STATUS_FAIL
        score = max([norm_acc(cv["acc"], 0.5) for cv in long_sig], default=0.0)
        best = max(ser, key=lambda r_: r_["linear_acc"])
        static = info["final_activity"] is not None and info["final_activity"] < 1e-6
        return TestResult(probe="memory", candidate_id=genome.genome_id, status=status, score=float(score),
                          p_value=float(min([cv["p_holm"] for cv in curve], default=1.0)),
                          effect_size=float(best.get("effect_size", float("nan"))),
                          details={"curve": curve, "horizon": max([cv["T"] for cv in any_sig], default=0),
                                   "leakage_detected": leak, "short_term_only": bool(any_sig and not long_sig),
                                   "static_storage": bool(static), "nonlinear_only": nonlin_only,
                                   "decoder": "ridge (primary), kNN (reported)", "mi_method": "plug-in + Miller-Madow"},
                          protocol={"S": S, "trials_per_class": n, "delays": delays, "history_steps": H, "split": split,
                                    "envs": [e.env_id for e in pool]}, split=split)

    # ------------------------------------------------------------------
    # LEVEL 4 — CANAL
    # ------------------------------------------------------------------
    def _channel_geometry(self, S: int):
        d = self.channel_distance(S)
        c = S // 2
        return d, (c, c - d // 2), (c, c - d // 2 + d)

    def probe_channel(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                      noise_level: float = 0.0, pre_damage: float = 0.0, S: Optional[int] = None,
                      env_change: Optional[Dict[str, Any]] = None) -> TestResult:
        n = trials or self.ps.trials
        S = S or self.cfg.world_size
        C = self.cfg.channels_max
        nsym = self.ps.channel_symbols
        rng = np.random.default_rng(derive_seed(seed, "channel"))
        r = self.ps.stim_radius
        pats = [self._pattern(rng, C, r) for _ in range(nsym)]
        mapping = rng.permutation(nsym)           # controle: correspondência símbolo→estímulo aleatorizada
        d, (ay, ax), (by, bx) = self._channel_geometry(S)
        pool = self.envs_for(split)
        n_null = max(2, n // 2)
        units, labels, is_null = [], [], []
        P = self.ps.stim_pulses
        for i in range(nsym * n + nsym * n_null):
            null = i >= nsym * n
            s = (i % nsym)
            pat = pats[mapping[0]] if null else pats[mapping[s]]
            units.append({"env": pool[(i // nsym) % len(pool)], "seed": derive_seed(seed, "ch", i), "group": i,
                          "stims": [(k + 1, pat, ay, ax, self.ps.stim_amp) for k in range(P)]})
            labels.append(s); is_null.append(null)
        times = sorted(set([0] + [t + 1 for t in self.ps.channel_times if t > 0]))
        damage = None
        if pre_damage > 0:
            y0 = max(0, ay - 2 * r); y1 = min(S, ay + 2 * r + 1)
            damage = {"box": (y0, y1, max(0, ax), min(S, bx + 1)), "frac": pre_damage, "recover": self.ps.recovery_steps}
        feats, info = self._run_protocol(genome, units, [{"name": "B", "kind": "region", "cy": by, "cx": bx,
                                                          "r": self.ps.obs_radius, "times": times}],
                                         max(times), seed, S=S, noise_level=noise_level, damage=damage,
                                         env_change=env_change)
        y = np.asarray(labels); nm = np.asarray(is_null)
        idx_s, idx_n = np.where(~nm)[0], np.where(nm)[0]
        ser = self._series(feats["B"], idx_s, y[idx_s], nsym, derive_seed(seed, "cs"), times)
        nul = self._series(feats["B"], idx_n, y[idx_n], nsym, derive_seed(seed, "cn"), times)
        pre = ser[0]
        post = [r_ for r_ in ser if r_["t"] > 0]
        sig = [r_ for r_ in post if r_["sig"]]
        leak = pre["sig"] or any(r_["sig"] for r_ in nul)
        best = max(post, key=lambda r_: r_["linear_acc"]) if post else pre
        latency = min([r_["t"] - 1 for r_ in sig], default=None)
        mi = best.get("mi_bits", 0.0)
        if any(info["bad"]):
            status = STATUS_ERROR
        elif leak:
            status = STATUS_INCONCLUSIVE
        elif sig:
            status = STATUS_PASS
        elif any(r_["nonlinear_only"] for r_ in post):
            status = STATUS_INCONCLUSIVE
        else:
            status = STATUS_FAIL
        return TestResult(probe="channel", candidate_id=genome.genome_id, status=status,
                          score=float(max(0.0, best["norm_acc"]) if sig else 0.0), p_value=float(best["p_holm"]),
                          effect_size=float(best.get("effect_size", float("nan"))),
                          details={"series": [{k: r_[k] for k in ["t", "linear_acc", "p_holm", "sig", "nonlinear_acc", "mi_bits"] if k in r_} for r_ in ser],
                                   "null_series": [{"t": r_["t"], "acc": r_["linear_acc"], "sig": r_["sig"]} for r_ in nul],
                                   "latency": latency, "distance": d, "mi_bits": mi,
                                   "capacity_bits_per_step": (mi / max(1, latency)) if latency is not None else 0.0,
                                   "confusion": best.get("confusion"), "leakage_detected": bool(leak),
                                   "noise_level": noise_level, "pre_damage": pre_damage, "mapping": mapping.tolist(),
                                   "best_t": best["t"]},
                          protocol={"S": S, "trials_per_symbol": n, "symbols": nsym, "distance": d, "times": times,
                                    "split": split, "A": (ay, ax), "B": (by, bx)}, split=split)

    # ------------------------------------------------------------------
    # LEVEL 5 — TRANSFORMAÇÃO
    # ------------------------------------------------------------------
    def probe_transformation(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                             S: Optional[int] = None, amp_scale: float = 1.0) -> TestResult:
        n = trials or self.ps.trials
        S = S or self.cfg.world_size
        C = self.cfg.channels_max
        rng = np.random.default_rng(derive_seed(seed, "transform"))
        r = self.ps.stim_radius
        P1, P2 = self._pattern(rng, C, r), self._pattern(rng, C, r)
        d, (ay, ax), (by, bx) = self._channel_geometry(S)
        off = max(r + 1, d // 3)
        a1c, a2c = ((ay - off) % S, ax), ((ay + off) % S, ax)
        pool = self.envs_for(split)
        combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        units, A1, A2 = [], [], []
        amp = self.ps.stim_amp * amp_scale
        for i in range(4 * n):
            a1, a2 = combos[i % 4]
            st_ = []
            for k in range(self.ps.stim_pulses):
                st_.append((k + 1, P1 * (1 if a1 else -1), a1c[0], a1c[1], amp))
                st_.append((k + 1, P2 * (1 if a2 else -1), a2c[0], a2c[1], amp))
            units.append({"env": pool[(i // 4) % len(pool)], "seed": derive_seed(seed, "tf", i), "group": i, "stims": st_})
            A1.append(a1); A2.append(a2)
        times = sorted(set([0] + [t + 1 for t in self.ps.transform_times]))
        feats, info = self._run_protocol(genome, units, [{"name": "B", "kind": "region", "cy": by, "cx": bx,
                                                          "r": self.ps.obs_radius, "times": times}], max(times), seed, S=S)
        A1, A2 = np.asarray(A1), np.asarray(A2)
        idx = np.arange(4 * n)
        targets = {"XOR": A1 ^ A2, "AND": A1 & A2, "OR": A1 | A2}
        res = {k: self._series(feats["B"], idx, v, 2, derive_seed(seed, k), times) for k, v in targets.items()}
        Xin = np.stack([2 * A1 - 1, 2 * A2 - 1], 1) + np.random.default_rng(seed).normal(0, 0.05, (4 * n, 2))
        base = decode_report(Xin, targets["XOR"], 2, self.ps.permutations, self.ps.folds, seed, nonlinear=False)
        xor_post = [r_ for r_ in res["XOR"] if r_["t"] > 0]
        sig = [r_ for r_ in xor_post if r_["sig"]]
        leak = res["XOR"][0]["sig"]
        base_sig = base["linear_p"] < self.th.alpha
        best = max(xor_post, key=lambda r_: r_["linear_acc"])
        if any(info["bad"]):
            status = STATUS_ERROR
        elif leak or base_sig:
            status = STATUS_INCONCLUSIVE
        elif sig:
            status = STATUS_PASS
        else:
            status = STATUS_FAIL
        return TestResult(probe="transformation", candidate_id=genome.genome_id, status=status,
                          score=float(max(0.0, best["norm_acc"]) if sig else 0.0), p_value=float(best["p_holm"]),
                          effect_size=float(best.get("effect_size", float("nan"))),
                          details={k: [{"t": r_["t"], "acc": r_["linear_acc"], "p_holm": r_["p_holm"], "sig": r_["sig"]} for r_ in v]
                                   for k, v in res.items()} | {"baseline_linear_xor_acc": base["linear_acc"],
                                                              "best_t": best["t"], "leakage_detected": bool(leak)},
                          protocol={"S": S, "trials_per_combo": n, "times": times, "split": split, "amp_scale": amp_scale},
                          split=split)

    # ------------------------------------------------------------------
    # LEVEL 6 — COMPOSIÇÃO
    # ------------------------------------------------------------------
    def probe_composition(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                          S: Optional[int] = None, t_mid: int = 8) -> TestResult:
        n = max(3, (trials or self.ps.trials) // 2)
        S = S or self.cfg.world_size
        C = self.cfg.channels_max
        rng = np.random.default_rng(derive_seed(seed, "compose"))
        r = self.ps.stim_radius
        P1, P2, P3 = (self._pattern(rng, C, r) for _ in range(3))
        d = int(max(2 * r + 2, min(self.channel_distance(S), (S - 4 * r) // 3)))
        c = S // 2
        ax = (c - (3 * d) // 2) % S
        mx, cx = (ax + d) % S, (ax + 2 * d) % S
        off = max(r + 1, d // 3)
        pool = self.envs_for(split)
        units, labs = [], []
        for i in range(8 * n):
            a1, a2, a3 = (i % 8) >> 2 & 1, (i % 8) >> 1 & 1, (i % 8) & 1
            st_ = []
            for k in range(self.ps.stim_pulses):
                st_.append((k + 1, P1 * (1 if a1 else -1), (c - off) % S, ax, self.ps.stim_amp))
                st_.append((k + 1, P2 * (1 if a2 else -1), (c + off) % S, ax, self.ps.stim_amp))
                st_.append((t_mid + k + 1, P3 * (1 if a3 else -1), c, mx, self.ps.stim_amp))
            units.append({"env": pool[(i // 8) % len(pool)], "seed": derive_seed(seed, "cp", i), "group": i, "stims": st_})
            labs.append((a1, a2, a3))
        labs = np.asarray(labs)
        times_c = sorted(set(t_mid + 1 + t for t in self.ps.transform_times))
        feats, info = self._run_protocol(genome, units,
                                         [{"name": "M", "kind": "region", "cy": c, "cx": mx, "r": self.ps.obs_radius, "times": [t_mid + 1]},
                                          {"name": "C", "kind": "region", "cy": c, "cx": cx, "r": self.ps.obs_radius, "times": times_c}],
                                         max(times_c), seed, S=S)
        idx = np.arange(8 * n)
        xor12 = labs[:, 0] ^ labs[:, 1]
        par = labs[:, 0] ^ labs[:, 1] ^ labs[:, 2]
        st1 = self._series(feats["M"], idx, xor12, 2, derive_seed(seed, "s1"))
        st2 = self._series(feats["C"], idx, par, 2, derive_seed(seed, "s2"))
        sig1 = any(r_["sig"] for r_ in st1)
        sig2 = [r_ for r_ in st2 if r_["sig"]]
        best = max(st2, key=lambda r_: r_["linear_acc"])
        if any(info["bad"]):
            status = STATUS_ERROR
        elif sig1 and sig2:
            status = STATUS_PASS
        elif sig2 and not sig1:
            status = STATUS_INCONCLUSIVE
        else:
            status = STATUS_FAIL
        return TestResult(probe="composition", candidate_id=genome.genome_id, status=status,
                          score=float(max(0.0, best["norm_acc"]) if (sig1 and sig2) else 0.0), p_value=float(best["p_holm"]),
                          effect_size=float(best.get("effect_size", float("nan"))),
                          details={"stage1_xor_at_M": [{"t": r_["t"], "acc": r_["linear_acc"], "sig": r_["sig"]} for r_ in st1],
                                   "stage2_parity_at_C": [{"t": r_["t"], "acc": r_["linear_acc"], "sig": r_["sig"]} for r_ in st2],
                                   "distance": d},
                          protocol={"S": S, "trials_per_combo": n, "t_mid": t_mid, "split": split}, split=split)

    # ------------------------------------------------------------------
    # LEVEL 7 — GENERALIZAÇÃO (entradas e ambientes não vistos)
    # ------------------------------------------------------------------
    def probe_generalization(self, genome: Genome, seed: int, trials: Optional[int] = None, S: Optional[int] = None,
                             task: str = "channel", split: str = "validation") -> TestResult:
        n = trials or self.ps.trials
        S = S or self.cfg.world_size
        C = self.cfg.channels_max
        rng = np.random.default_rng(derive_seed(seed, "gen"))
        r = self.ps.stim_radius
        d, (ay, ax), (by, bx) = self._channel_geometry(S)
        train_envs, test_envs = self.envs_for("evolution"), self.envs_for(split)
        nsym = self.ps.channel_symbols if task == "channel" else 4
        pats = [self._pattern(rng, C, r) for _ in range(max(nsym, 2))]
        units, labels, is_test = [], [], []
        for i in range(nsym * n * 2):
            test = i >= nsym * n
            s = i % nsym
            amp = self.ps.stim_amp * (rng.choice([0.6, 1.4]) if test else 1.0)
            jy, jx = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))) if test else (0, 0)
            env = (test_envs if test else train_envs)[(i // nsym) % len(test_envs if test else train_envs)]
            st_ = []
            for k in range(self.ps.stim_pulses):
                if task == "channel":
                    st_.append((k + 1, pats[s], ay + jy, ax + jx, amp))
                else:
                    a1, a2 = s >> 1 & 1, s & 1
                    off = max(r + 1, d // 3)
                    st_.append((k + 1, pats[0] * (1 if a1 else -1), (ay - off + jy) % S, ax + jx, amp))
                    st_.append((k + 1, pats[1] * (1 if a2 else -1), (ay + off + jy) % S, ax + jx, amp))
            units.append({"env": env, "seed": derive_seed(seed, "gn", i, "test" if test else "train"), "group": i, "stims": st_})
            labels.append(s if task == "channel" else ((s >> 1 & 1) ^ (s & 1)))
            is_test.append(test)
        times = sorted(set(t + 1 for t in (self.ps.channel_times if task == "channel" else self.ps.transform_times) if t > 0))
        feats, info = self._run_protocol(genome, units, [{"name": "B", "kind": "region", "cy": by, "cx": bx,
                                                          "r": self.ps.obs_radius, "times": times}], max(times), seed, S=S)
        y = np.asarray(labels); te = np.asarray(is_test)
        ncls = nsym if task == "channel" else 2
        tr_idx, te_idx = np.where(~te)[0], np.where(te)[0]
        tr_ser = self._series(feats["B"], tr_idx, y[tr_idx], ncls, derive_seed(seed, "gtr"), times)
        best = max(tr_ser, key=lambda r_: r_["linear_acc"])     # t* escolhido apenas no treino (sem p-hacking)
        t_star = best["t"]
        X = feats["B"][t_star]
        yp = ridge_predict(X[tr_idx], y[tr_idx], X[te_idx], ncls)
        acc_te = balanced_accuracy(y[te_idx], yp, ncls)
        k = int((yp == y[te_idx]).sum())
        p_te = binom_p_greater(k, len(te_idx), 1.0 / ncls)
        tr_norm, te_norm = best["norm_acc"], norm_acc(acc_te, 1.0 / ncls)
        if any(info["bad"]):
            status, label = STATUS_ERROR, "ERROR"
        elif not best["sig"]:
            status, label = STATUS_FAIL, "NO_TRAIN_SIGNAL"
        elif p_te < self.th.alpha and te_norm >= self.th.min_norm_acc and (tr_norm - te_norm) <= self.th.generalization_drop:
            status, label = STATUS_PASS, "GENERALIZES"
        else:
            status, label = STATUS_FAIL, "OVERFIT"
        return TestResult(probe="generalization", candidate_id=genome.genome_id, status=status,
                          score=float(max(0.0, te_norm)) if status == STATUS_PASS else 0.0, p_value=float(p_te),
                          effect_size=float(te_norm - 0.0),
                          details={"label": label, "task": task, "t_star": t_star, "train_norm_acc": tr_norm,
                                   "test_norm_acc": te_norm, "test_acc": acc_te, "n_test": int(len(te_idx)),
                                   "unseen": "amplitudes {0.6,1.4}, jitter ±1, sementes novas, ambientes de " + split},
                          protocol={"S": S, "trials": n, "task": task, "split": split, "times": times}, split=split)

    # ------------------------------------------------------------------
    # AUTORREPARAÇÃO / RECUPERAÇÃO
    # ------------------------------------------------------------------
    def _largest_structure(self, U: torch.Tensor) -> Tuple[np.ndarray, bool]:
        mag = U.pow(2).sum(0).sqrt().cpu().numpy()
        thr = mag.mean() + mag.std()
        mask = (mag > thr) & (mag > 1e-3)
        if HAVE_SCIPY:
            lab, nlab = sndi.label(mask)
            if nlab == 0:
                return mask, False
            sizes = sndi.sum(mask, lab, index=np.arange(1, nlab + 1))
            big = int(np.argmax(sizes)) + 1
            comp = lab == big
        else:
            comp = mask
        return comp, bool(comp.sum() >= 4)

    def _similarity(self, Ud: torch.Tensor, Ur: torch.Tensor, region: torch.Tensor) -> Dict[str, float]:
        md = Ud.pow(2).sum(0).sqrt(); mr = Ur.pow(2).sum(0).sqrt()
        thr = mr.mean() + torch.clamp(mr.std(), min=0.05 * float(mr.mean()) + 1e-6)
        a, b = md > thr, mr > thr
        un = (a | b).float().sum().item()
        iou = 1.0 if un == 0 else (a & b).float().sum().item() / un
        reg = region.float()
        vd, vr = (Ud * reg).flatten(), (Ur * reg).flatten()
        cos = float(torch.dot(vd, vr) / (vd.norm() * vr.norm() + 1e-12)) if vr.norm() > 1e-9 else float(vd.norm() < 1e-9)
        traj = 1.0 - float((Ud - Ur).norm() / (Ud.norm() + Ur.norm() + 1e-12))
        return {"geometry": float(iou), "internal": max(0.0, cos), "trajectory": traj,
                "similarity": float((iou + max(0.0, cos) + traj) / 3)}

    def probe_recovery(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None,
                       S: Optional[int] = None, allow_fallback: bool = False, levels: Optional[List[float]] = None,
                       envs: Optional[List[Environment]] = None) -> TestResult:
        n = max(2, (trials or self.ps.trials) // 2)
        levels = levels or list(self.ps.damage_levels)
        gb, st, prot = self._setup(genome, n, split, seed, S=S, envs=envs)
        S = gb.S
        st = self.sim.run(gb, st, self.ps.recovery_warmup)
        regions, found = [], []
        for i in range(n):
            comp, ok = self._largest_structure(st.U[i])
            if not ok and allow_fallback:
                comp = np.zeros((S, S), bool); comp[S // 4:3 * S // 4, S // 4:3 * S // 4] = True
                ok = True
            regions.append(comp); found.append(ok)
        if not any(found):
            return TestResult(probe="recovery", candidate_id=genome.genome_id, status=STATUS_INCONCLUSIVE, score=0.0,
                              details={"note": "nenhuma estrutura persistente identificável"}, protocol=prot, split=split)
        L = len(levels)
        idx, groups = [], []
        for i in range(n):
            idx += [i] * (L + 1); groups += [i] * (L + 1)
        gb2 = gb.select(idx)
        st2 = st.select(idx, groups=groups, seed=derive_seed(seed, "branch"))
        rng = np.random.default_rng(derive_seed(seed, "dmg"))
        mask = np.zeros((len(idx), 1, S, S), bool)
        for i in range(n):
            if not found[i]:
                continue
            cells = np.argwhere(regions[i])
            for j, p in enumerate(levels):
                k = int(round(p * len(cells)))
                sel = cells[rng.choice(len(cells), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
                mask[i * (L + 1) + j + 1, 0, sel[:, 0], sel[:, 1]] = True
        st2.damage_cells(torch.as_tensor(mask, device=self.sim.device))
        dil = []
        for i in range(n):
            reg = regions[i]
            if HAVE_SCIPY:
                reg = sndi.binary_dilation(reg, iterations=2)
            dil.append(torch.as_tensor(reg, device=self.sim.device).unsqueeze(0))

        def measure():
            out = np.zeros((n, L, 4))
            for i in range(n):
                ref = st2.U[i * (L + 1)]
                for j in range(L):
                    s_ = self._similarity(st2.U[i * (L + 1) + j + 1], ref, dil[i])
                    out[i, j] = [s_["geometry"], s_["internal"], s_["trajectory"], s_["similarity"]]
            return out
        sim0 = measure()
        st2 = self.sim.run(gb2, st2, self.ps.recovery_steps)
        sim1 = measure()
        rec = (sim1[..., 3] - sim0[..., 3]) / (1 - sim0[..., 3] + 1e-6)
        rec = np.clip(rec, -1, 1)
        ok_rows = np.asarray(found)
        curve = [{"damage": float(p), "sim_immediate": float(sim0[ok_rows, j, 3].mean()),
                  "sim_recovered": float(sim1[ok_rows, j, 3].mean()), "recovery": float(rec[ok_rows, j].mean()),
                  "geometry": float(sim1[ok_rows, j, 0].mean()), "internal": float(sim1[ok_rows, j, 1].mean()),
                  "trajectory": float(sim1[ok_rows, j, 2].mean())} for j, p in enumerate(levels)]
        low = [cv for cv in curve if cv["damage"] <= 0.5]
        mean_rec = float(np.mean([cv["recovery"] for cv in low]))
        mean_sim = float(np.mean([cv["sim_recovered"] for cv in low]))
        if bool(st2.bad.any()):
            status = STATUS_ERROR
        elif mean_rec >= self.th.recovery_gain and mean_sim >= self.th.recovery_sim:
            status = STATUS_PASS
        else:
            status = STATUS_FAIL
        lo, hi = bootstrap_ci(rec[ok_rows][:, [j for j, p in enumerate(levels) if p <= 0.5]].ravel())
        return TestResult(probe="recovery", candidate_id=genome.genome_id, status=status, score=max(0.0, mean_rec),
                          ci=(lo, hi), details={"curve": curve, "mean_recovery_le50": mean_rec,
                                                "mean_similarity_le50": mean_sim, "structures_found": int(sum(found)),
                                                "fallback_region": bool(allow_fallback)},
                          protocol={**prot, "levels": levels, "warmup": self.ps.recovery_warmup,
                                    "recovery_steps": self.ps.recovery_steps}, split=split)

    # ------------------------------------------------------------------
    # ROBUSTEZ
    # ------------------------------------------------------------------
    def _variant(self, genome: Genome, fn: Callable[[Dict[str, np.ndarray]], None], tag: str) -> Genome:
        p = {k: np.array(v, copy=True) for k, v in genome.params.items()}
        fn(p)
        return Genome(params=self.space.clamp(p), genome_id=f"{genome.genome_id}~{tag}", parent_ids=[genome.genome_id],
                      generation=genome.generation, origin="test-variant")

    def _desc_runs(self, genomes: List[Genome], envs: List[Environment], seed: int, S: Optional[int] = None,
                   hook: Optional[Callable] = None) -> Tuple[np.ndarray, List[str]]:
        rows = []
        for i, g in enumerate(genomes):
            env = envs[i % len(envs)]
            rows.append((g, env, derive_seed(seed, "rob", i)))
        gl = [r_[0] for r_ in rows]
        # reutiliza o pipeline de triagem com sementes fixas por réplica
        out = []
        per = max(1, self.cfg.batch_size // 2)
        for c0 in range(0, len(rows), per):
            chunk = rows[c0:c0 + per]
            jobs = [(j, e, s) for j, (_, e, s) in enumerate(chunk)]
            res = self.screener._screen_chunk([g for g, _, _ in chunk], jobs, self.ps.robust_steps,
                                              S or self.cfg.world_size, derive_seed(seed, "rob-run", c0), hook)
            out.extend(res)
        vecs = np.stack([descriptor_vector(r_) for r_ in out])
        via = [classify_viability(r_, self.th, r_["bad"]) for r_ in out]
        return vecs, via

    def probe_robustness(self, genome: Genome, seed: int, split: str = "evolution", trials: Optional[int] = None) -> TestResult:
        n = trials or self.ps.robust_trials
        envs = self.envs_for(split)
        base_v, base_via = self._desc_runs([genome] * n, envs, seed)
        v0 = base_v.mean(0)
        dim = math.sqrt(len(v0))

        grand = self.sim._grand

        def noise_hook(level):
            def h(st):
                st.U = st.U + level * grand(st, st.U.shape[1:])
            return h

        def sp_hook(frac):
            def h(st):
                if st.t % 8 == 0:
                    m = grand(st, (1,) + tuple(st.U.shape[2:]), "uniform") < frac
                    sgn = torch.sign(grand(st, st.U.shape[1:], "uniform") - 0.5)
                    st.U = torch.where(m, sgn * 0.9 * self.cfg.state_bound / 3, st.U)
            return h

        def del_hook(frac):
            def h(st):
                if st.t % 8 == 0:
                    st.damage_cells(grand(st, (1,) + tuple(st.U.shape[2:]), "uniform") < frac)
            return h

        drift_rng = np.random.default_rng(derive_seed(seed, "drift"))

        def drift(p):
            for nme, s in self.space.specs.items():
                if s.kind in ("float", "logfloat") and s.group != "mutation_parameters":
                    p[nme] = np.asarray(p[nme]) * (1 + drift_rng.normal(0, 0.05, size=np.shape(p[nme])))

        conds = []
        for lvl in [l for l in self.ps.noise_levels if l > 0]:
            conds.append(("gaussian_noise", lvl, [genome] * n, envs, None, noise_hook(lvl)))
        conds.append(("salt_pepper", 0.01, [genome] * n, envs, None, sp_hook(0.01)))
        conds.append(("cell_deletion", 0.02, [genome] * n, envs, None, del_hook(0.02)))
        conds.append(("parameter_drift", 0.05, [self._variant(genome, drift, "drift")] * n, envs, None, None))
        for f_ in [0.8, 1.25]:
            conds.append(("time_step_variation", f_, [self._variant(genome, lambda p, f_=f_: p.__setitem__("time_scale", np.float32(float(p["time_scale"]) * f_)), f"dt{f_}")] * n, envs, None, None))
        for b in range(len(BOUNDARIES)):
            if b != int(genome.params["boundary"]):
                conds.append(("boundary_variation", BOUNDARIES[b], [self._variant(genome, lambda p, b=b: p.__setitem__("boundary", np.int64(b)), f"b{b}")] * n, envs, None, None))
        other = self.envs_for("validation") if split == "evolution" else self.envs_for("evolution")
        conds.append(("initial_state_variation", other[0].env_id, [genome] * n, other, None, None))
        S2 = int(round(self.cfg.world_size * 1.5 / 2) * 2)
        conds.append(("world_size_variation", S2, [genome] * n, envs, S2, None))
        rows = []
        for name, level, gl, ev, S_, hook in conds:
            v, via = self._desc_runs(gl, ev, seed, S=S_, hook=hook)
            sim_ = float(1 - np.linalg.norm(v.mean(0) - v0) / dim)
            viable = float(np.mean([x == "VIABLE" for x in via]))
            rows.append({"condition": name, "level": level, "similarity": sim_, "viable_fraction": viable,
                         "ok": bool(viable >= 0.5 and sim_ >= self.th.robust_sim)})
        score = float(np.mean([r_["ok"] for r_ in rows]))
        status = STATUS_PASS if score >= self.th.robustness else STATUS_FAIL
        curve = [{"noise": 0.0, "similarity": 1.0}] + [{"noise": r_["level"], "similarity": r_["similarity"]}
                                                       for r_ in rows if r_["condition"] == "gaussian_noise"]
        return TestResult(probe="robustness", candidate_id=genome.genome_id, status=status, score=score,
                          details={"conditions": rows, "noise_curve": curve,
                                   "baseline_viable": float(np.mean([x == "VIABLE" for x in base_via]))},
                          protocol={"trials": n, "split": split, "steps": self.ps.robust_steps}, split=split)

    # ------------------------------------------------------------------
    # LEVEL 8 — ADAPTAÇÃO
    # ------------------------------------------------------------------
    def probe_adaptation(self, genome: Genome, seed: int, split: str = "validation", trials: Optional[int] = None) -> TestResult:
        n = trials or self.ps.trials
        gb, st, prot = self._setup(genome, n, split, seed, repeat=2)
        S = gb.S
        m = min(self.cfg.metric_res, S)
        W = max(16, self.ps.adaptation_steps // 2)
        st = self.sim.run(gb, st, self.ps.warmup)

        def window(st):
            frames = []
            for t in range(W):
                st = self.sim.step(gb, st)
                if t % 2 == 1:
                    frames.append(F.adaptive_avg_pool2d(st.U, m))
            Fr = torch.stack(frames)
            d = compute_descriptors(Fr, None, None, gb.chm.view(gb.B, -1), gb.sat.view(gb.B, -1), S, 2, W)
            vec = np.stack([descriptor_vector({k: float(v[i]) for k, v in d.items()} | {"lyapunov": 0.0, "causal_spread": 0.0})
                            for i in range(gb.B)])
            return st, vec
        st, pre = window(st)
        units = [{"changed": i >= n} for i in range(2 * n)]
        self._apply_env_change(st, {"resource_mult": 0.5, "noise_add": 0.01}, units)
        st = self.sim.run(gb, st, self.ps.adaptation_steps)
        st, post = window(st)
        change = np.linalg.norm(post - pre, axis=1)
        p_change = mann_whitney_greater(change[n:], change[:n])
        func = self.probe_channel(genome, derive_seed(seed, "adapt-func"), split=split, trials=trials,
                                  env_change={"t": -1, "resource_mult": 0.5, "noise_add": 0.01})
        behav_ok = p_change < self.th.adaptation_p
        func_ok = func.status == STATUS_PASS
        status = STATUS_PASS if (behav_ok and func_ok) else STATUS_FAIL
        return TestResult(probe="adaptation", candidate_id=genome.genome_id, status=status,
                          score=float(func.score if status == STATUS_PASS else 0.0), p_value=float(p_change),
                          effect_size=cohens_d(change[n:], change[:n]),
                          details={"behavior_change_changed": float(change[n:].mean()),
                                   "behavior_change_control": float(change[:n].mean()), "behavior_p": p_change,
                                   "function_status": func.status, "function_score": func.score,
                                   "interpretation": "mudança de comportamento acima da deriva natural mantendo canal funcional; "
                                                     "não implica aprendizado, vida ou agência"},
                          protocol={**prot, "window": W, "change": {"resource_mult": 0.5, "noise_add": 0.01}}, split=split)

    # ------------------------------------------------------------------
    # REPLICAÇÃO / ABLAÇÃO / TRIAGEM LONGA
    # ------------------------------------------------------------------
    def probe_replication(self, genome: Genome, seed: int, target: str = "memory", n_seeds: Optional[int] = None,
                          split: str = "validation") -> TestResult:
        n_seeds = n_seeds or self.ps.replication_seeds
        runs = [self.run_probe(target, genome, derive_seed(seed, "rep", i), split=split) for i in range(n_seeds)]
        scores = np.asarray([r_.score if np.isfinite(r_.score) else 0.0 for r_ in runs])
        passes = np.asarray([r_.status == STATUS_PASS for r_ in runs])
        rate = float(passes.mean())
        lo, hi = bootstrap_ci(scores)
        status = STATUS_PASS if rate >= self.th.replication_pass_rate else (STATUS_INCONCLUSIVE if rate >= 0.5 else STATUS_FAIL)
        return TestResult(probe="replication", candidate_id=genome.genome_id, status=status, score=rate,
                          p_value=binom_p_greater(int(passes.sum()), n_seeds, self.th.alpha),
                          effect_size=cohens_d(scores, np.zeros_like(scores) + 1e-9), ci=(lo, hi),
                          details={"target": target, "pass_rate": rate, "scores": scores.tolist(),
                                   "statuses": [r_.status for r_ in runs], "stats": summary_stats(scores)},
                          protocol={"target": target, "n_seeds": n_seeds, "split": split}, split=split)

    ABLATIONS = {
        "remove_diffusion": lambda p: p.__setitem__("diff", np.zeros_like(p["diff"])),
        "remove_long_range": lambda p: p.__setitem__("link_active", np.zeros_like(p["link_active"])),
        "remove_conservation": lambda p: p.__setitem__("conserve", np.zeros_like(p["conserve"])),
        "remove_reaction": lambda p: p.__setitem__("Q", np.zeros_like(p["Q"])),
        "remove_inertia": lambda p: p.__setitem__("inertia", np.zeros_like(p["inertia"])),
        "remove_delay": lambda p: p.__setitem__("delay_w", np.zeros_like(p["delay_w"])),
        "remove_hysteresis": lambda p: p.__setitem__("hyst_w", np.zeros_like(p["hyst_w"])),
        "remove_resource_limit": lambda p: p.__setitem__("res_enabled", np.zeros_like(p["res_enabled"])),
        "remove_plasticity": lambda p: p.__setitem__("link_eta", np.zeros_like(p["link_eta"])),
        "remove_noise": lambda p: p.__setitem__("noise", np.zeros_like(p["noise"])),
        "remove_advection": lambda p: p.__setitem__("vel", np.zeros_like(p["vel"])),
        "remove_nonlinear_growth": lambda p: p.__setitem__("g_gain", np.zeros_like(p["g_gain"])),
    }

    def probe_ablation(self, genome: Genome, seed: int, target: str = "memory", split: str = "validation") -> TestResult:
        base = self.run_probe(target, genome, seed, split=split)
        variants = []
        for name, fn in self.ABLATIONS.items():
            v = self._variant(genome, fn, name)
            if v.hash() != self._variant(genome, lambda p: None, "id").hash():
                variants.append((name, v))
        act = np.where(genome.params["ch_active"])[0]
        if len(act) > 1:
            for c in act:
                variants.append((f"remove_channel_{c}", self._variant(genome, lambda p, c=c: p["ch_active"].__setitem__(c, False), f"rc{c}")))
        kact = np.where(genome.params["k_active"])[0]
        if len(kact) > 1:
            for k in kact:
                variants.append((f"remove_kernel_{k}", self._variant(genome, lambda p, k=k: p["k_active"].__setitem__(k, False), f"rk{k}")))
        rows = []
        bscore = base.score if np.isfinite(base.score) else 0.0
        for name, v in variants:
            r_ = self.run_probe(target, v, seed, split=split)
            sc = r_.score if np.isfinite(r_.score) else 0.0
            rows.append({"ablation": name, "status": r_.status, "score": sc, "importance": float(bscore - sc)})
        rows.sort(key=lambda x: -x["importance"])
        return TestResult(probe="ablation", candidate_id=genome.genome_id, status=STATUS_PASS if rows else STATUS_INCONCLUSIVE,
                          score=float(max([r_["importance"] for r_ in rows], default=0.0)),
                          details={"target": target, "base_status": base.status, "base_score": bscore, "ablations": rows,
                                   "essential": [r_["ablation"] for r_ in rows if r_["importance"] > 0.5 * max(bscore, 1e-9) and bscore > 0]},
                          protocol={"target": target, "split": split}, split=split)

    def probe_long_screen(self, genome: Genome, seed: int, split: str = "evolution") -> TestResult:
        """Avaliação longa para sobreviventes (multi-fidelidade): mais passos, mundo maior."""
        S2 = int(round(self.cfg.world_size * 1.5 / 2) * 2)
        d = self.screener.screen([genome], self.envs_for(split), seed, steps=self.cfg.screen_steps * 2, S=S2,
                                 n_envs=len(self.envs_for(split)))[0]
        ok = d["viability"] == "VIABLE"
        return TestResult(probe="long_screen", candidate_id=genome.genome_id, status=STATUS_PASS if ok else STATUS_FAIL,
                          score=float(d["persistence"]), details={k: v for k, v in d.items() if not isinstance(v, list)},
                          protocol={"S": S2, "steps": self.cfg.screen_steps * 2, "split": split}, split=split)
