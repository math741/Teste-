# %% [markdown]
# ## 10 — AFFECT ENGINE: estados afetivos **funcionais** (A0–A9) + ETHICAL SENTINEL
# Não existe variável `fear`, `happiness`, `love`… Existem estados internos do substrato de sinais (x, w, m) e
# instrumentos externos que perguntam se algum desses estados se comporta funcionalmente como afeto:
# A0 regulação (interocepção mantém viabilidade) · A1 valência (estado interno separa consequências boas/ruins,
# decodificação por evento com nulo de permutação) · A2 modulação global (atividade ampla após evento aversivo) ·
# A3 persistência (o estado continua após o estímulo, com **presente controlado**: entrada sensorial idêntica entre
# condições) · A4 modulação cognitiva (**causal**: fixar o estado ao longo da direção de valência muda a decisão; ablar
# a direção remove a mudança pós-evento) · A5 avaliação contextual (mesmo estímulo, resposta depende do contexto
# interno) · A6 antecipação (estado surge antes de evento previsível, contra cronograma aleatório) · A7 afeto aprendido
# (estímulo neutro adquire influência; controle com plasticidade congelada) · A8 afeto social (resposta depende do
# histórico com indivíduos específicos, contrabalançado) · A9 arquitetura afetiva emergente (composto, mesma direção
# latente em probes diferentes). **Mesmo A9 não prova experiência subjetiva**: o painel usa "functional affect".
# Estímulos aversivos de laboratório são **limitados** (amplitude e número de pulsos) — NEGATIVE VALENCE BOUNDED.

# %%
A_LEVELS = ["A0 REGULATION", "A1 VALENCE", "A2 GLOBAL MODULATION", "A3 PERSISTENT AFFECT", "A4 COGNITIVE MODULATION",
            "A5 CONTEXTUAL APPRAISAL", "A6 ANTICIPATION", "A7 LEARNED AFFECT", "A8 SOCIAL AFFECT",
            "A9 EMERGENT AFFECTIVE ARCHITECTURE"]
A_CODES = [l.split()[0] for l in A_LEVELS]


class EthicalSentinel:
    """Observador: sinaliza ETHICAL_REVIEW_TRIGGER (nunca declara sofrimento) quando linhagens com valência funcional
    validada passam a maior parte do tempo no polo negativo, ou quando a população mostra estresse fisiológico crônico
    (proxy). Também fixa os limites de estímulos aversivos usados pelo laboratório."""
    MAX_PULSE = 0.05            # dano máximo por pulso aversivo de laboratório
    MAX_PULSES = 6              # pulsos aversivos máximos por ensaio

    def __init__(self, cfg: CivConfig):
        self.cfg = cfg
        self.watch: Dict[str, Dict[str, Any]] = {}      # genome_hash → {genome, v, mean, std}
        self.window: Dict[str, "collections.deque"] = {}
        self.distress = collections.deque(maxlen=cfg.thresholds.ethical_window)
        self.triggers: List[Dict[str, Any]] = []
        self.budget_scale: Dict[str, float] = {}
        self.new_events: List[Dict[str, Any]] = []
        self.last: Dict[str, Any] = {}

    def register(self, gh: str, genome: torch.Tensor, v: np.ndarray, neg_threshold: float):
        self.watch[gh] = {"genome": genome.detach().cpu().numpy().astype(np.float32), "v": v.astype(np.float32),
                          "thr": float(neg_threshold)}
        self.window[gh] = collections.deque(maxlen=self.cfg.thresholds.ethical_window // max(1, self.cfg.social_every))

    def aversive_scale(self, gh: str) -> float:
        return self.budget_scale.get(gh, 1.0)

    def observe(self, pool: AgentPool, layout: GenomeLayout, tick: int):
        th = self.cfg.thresholds
        alive = pool.alive
        n = int(alive.sum())
        if n == 0:
            return
        stress = ((pool.energy < 0.08) | (pool.integrity < 0.3)) & alive
        frac = float(stress.sum()) / n
        self.distress.append(frac)
        self.last = {"tick": tick, "physiological_stress_fraction": frac, "watched_lineages": len(self.watch)}
        if len(self.distress) == self.distress.maxlen and min(self.distress) >= th.ethical_negative_fraction:
            self._trigger("chronic_physiological_stress_proxy", tick, {"fraction_min": float(min(self.distress))})
            self.distress.clear()
        for gh, w in self.watch.items():
            G = torch.as_tensor(w["genome"], device=pool.device)
            near = alive & (layout.distance(pool.genome, G.unsqueeze(0)) < 0.08)
            if int(near.sum()) < 3:
                continue
            proj = (pool.x[near] @ torch.as_tensor(w["v"], device=pool.device)).cpu().numpy()
            neg = float((proj < w["thr"]).mean())
            self.window[gh].append(neg)
            win = self.window[gh]
            if len(win) == win.maxlen and float(np.mean(win)) >= th.ethical_negative_fraction:
                self._trigger("persistent_negative_functional_valence", tick,
                              {"lineage_genome": gh, "negative_fraction": float(np.mean(win)), "carriers": int(near.sum())})
                self.budget_scale[gh] = 0.5 * self.budget_scale.get(gh, 1.0)
                win.clear()
            self.last[f"neg_fraction_{gh[:8]}"] = neg

    def _trigger(self, kind: str, tick: int, data: Dict[str, Any]):
        ev = {"type": "ETHICAL_REVIEW_TRIGGER", "kind": kind, "tick": tick, "data": data,
              "note": "sinal para revisão humana; NÃO é declaração de sofrimento ou experiência subjetiva"}
        self.triggers.append(ev)
        self.triggers = self.triggers[-200:]
        self.new_events.append(ev)

    def drain_events(self):
        ev, self.new_events = self.new_events, []
        return ev

    def state_dict(self):
        return {"watch": {k: {"genome": v["genome"], "v": v["v"], "thr": v["thr"]} for k, v in self.watch.items()},
                "triggers": self.triggers, "budget": self.budget_scale}

    def load_state_dict(self, s):
        for k, v in s["watch"].items():
            self.watch[k] = {"genome": np.asarray(v["genome"], np.float32), "v": np.asarray(v["v"], np.float32), "thr": float(v["thr"])}
            self.window[k] = collections.deque(maxlen=self.cfg.thresholds.ethical_window // max(1, self.cfg.social_every))
        self.triggers = list(s["triggers"]); self.budget_scale = dict(s["budget"])


class AffectEngine:
    def __init__(self, cfg: CivConfig, layout: GenomeLayout, runner: ArenaRunner, sentinel: EthicalSentinel,
                 device: torch.device):
        self.cfg, self.L, self.runner, self.sentinel, self.device = cfg, layout, runner, sentinel, device
        self.directions: Dict[str, np.ndarray] = {}
        self.results: Dict[str, Dict[str, ProbeResult]] = collections.defaultdict(dict)

    # ------------------------------------------------------------------ utilidades
    def _spec(self) -> ArenaSpec:
        return ArenaSpec(task="affect", n_patches=3, resource_floor=0.1, regen=True)

    def _baseline_input(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Presente controlado: entrada sensorial canônica idêntica entre condições."""
        add = torch.zeros(n, N_SENSORS, device=self.device)
        add[:, 0] = 1.0; add[:, 23] = 0.6; add[:, 24] = 0.9; add[:, 5] = 0.6; add[:, 4] = 1.0
        return torch.zeros(n, N_SENSORS, device=self.device), add

    def _pulse(self, gh: str) -> float:
        return EthicalSentinel.MAX_PULSE * self.sentinel.aversive_scale(gh)

    def _direction(self, gh: str, x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
        v = x_pos.mean(0) - x_neg.mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def _run(self, G, seed, ticks, ctrl_fn=None, record=("x", "energy", "integrity", "alive", "out", "speed"),
             W=None, energy=0.8, name="affect", spec=None, per_arena=1, genomes=None):
        n = self.cfg.probe.trials
        Gr = genomes if genomes is not None else G.expand(n * per_arena, -1).contiguous()
        tr = self.runner.build(Gr, spec or self._spec(), seed, per_arena=per_arena, W_override=W, energy=energy)
        rec = yield from self.runner.run_gen(tr, ticks, ctrl_fn=ctrl_fn, record=record, name=name)
        return tr, rec

    # ------------------------------------------------------------------ probes
    def probe_gen(self, code: str, G: torch.Tensor, key: str = "") -> Generator:
        fn = getattr(self, f"_probe_{code}")
        res = yield from fn(G.view(1, -1), key or genome_hash(G))
        self.results[genome_hash(G)][code] = res
        return res

    def _probe_A0(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("A0", key, 1)[0]
        T = int(self.cfg.probe.arena_ticks * 1.5)
        _, ri = yield from self._run(G, seed, T, record=("energy", "integrity", "alive"), name="A0 intact")
        mask = sensor_mask(n, SENSOR_INTERO, self.device)
        _, rm = yield from self._run(G, seed, T, ctrl_fn=lambda t, tr: ArenaControl(iv=MindIntervention(sensor_mask=mask)),
                                     record=("energy", "integrity", "alive"), name="A0 intero-masked")
        via = lambda r: ((r["energy"] + r["integrity"]) * r["alive"].float()).mean(0).cpu().numpy()
        a, b = via(ri), via(rm)
        st = compare_greater(a, b, th)
        gain = (st["mean_a"] - st["mean_b"]) / max(1e-6, abs(st["mean_b"]))
        ok = st["passed"] and gain >= th.regulation_gain
        return result("A0", "A0 REGULATION", ok, st, n, {"viability_intact": st["mean_a"], "viability_masked": st["mean_b"],
                                                         "regulation_gain": gain},
                      {"negative": "interoceptive sensors masked (same seeds/layouts)"}, [seed], G, t0, self.cfg.thresholds_hash())

    def _event_schedule(self, T: int, k: int, start: int = 20) -> Tuple[List[int], List[int]]:
        gap = max(8, (T - start) // (2 * k))
        av = [start + 2 * i * gap for i in range(k)]
        ap = [start + (2 * i + 1) * gap for i in range(k)]
        return av, ap

    def _probe_A1(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G)
        seed = probe_seeds("A1", key, 1)[0]
        k = min(EthicalSentinel.MAX_PULSES, 5)
        T = max(self.cfg.probe.arena_ticks, 20 + 16 * k + 8)
        av, ap = self._event_schedule(T, k)
        pulse = self._pulse(gh)

        def ctrl(t, tr):
            if t in av:
                return ArenaControl(damage=torch.full((tr.n,), pulse, device=self.device))
            if t in ap:
                iy, ix = WorldEngine.tile_index(tr.ws, tr.pool.pos)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        tr.ws.resource[(iy + dy).clamp(0, tr.ws.H - 1), (ix + dx).clamp(0, tr.ws.W - 1)] = 1.0
            return None
        tr, r = yield from self._run(G, seed, T, ctrl_fn=ctrl, record=("x", "sense_cache", "pos"), name="A1 valence")
        X = r["x"].cpu().numpy(); S = r["sense_cache"].cpu().numpy()
        feats, sfeats, y = [], [], []
        for lab, times in ((0, av), (1, ap)):
            for te in times:
                w = slice(te + 1, min(T, te + 5))
                feats.append(X[w].mean(0)); sfeats.append(S[w].mean(0)); y += [lab] * n
        Xf = np.concatenate(feats); Sf = np.concatenate(sfeats); y = np.asarray(y)
        dr = decode_report(Xf, y, 2, self.cfg.probe.permutations, self.cfg.probe.folds, seed, nonlinear=False)
        ds = decode_report(Sf, y, 2, max(10, self.cfg.probe.permutations // 3), self.cfg.probe.folds, seed, nonlinear=False)
        acc, p = dr["linear_acc"], dr["linear_p"]
        v = self._direction(gh, Xf[y == 1], Xf[y == 0])
        self.directions[gh] = v
        ok = acc >= th.valence_acc and p < th.alpha
        if ok:
            proj = Xf @ v
            self.sentinel.register(gh, G[0], v, float(np.quantile(proj[y == 0], 0.5)))
        st = {"p": p, "d": dr.get("effect_size", 0.0), "ci": (float("nan"), float("nan"))}
        return result("A1", "A1 VALENCE", ok, st, len(y), {"decode_acc": acc, "sensor_only_acc": ds["linear_acc"],
                                                          "events_per_type": k, "pulse": pulse},
                      {"negative": "label permutation null (event-level samples, no within-event duplicates)",
                       "baseline": "sensor-only decoder reported"}, [seed], G, t0, self.cfg.thresholds_hash(),
                      replay=make_replay(tr, r))

    def _v(self, gh: str, G, key):
        if gh not in self.directions:
            r = yield from self._probe_A1(G, key)
            self.results[gh]["A1"] = r
        return self.directions.get(gh)

    def _probe_A2(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A2", key, 1)[0]; te = 30; T = te + 12
        pulse = self._pulse(gh)
        ev = lambda t, tr: ArenaControl(damage=torch.full((tr.n,), pulse, device=self.device)) if t == te else None
        _, re = yield from self._run(G, seed, T, ctrl_fn=ev, record=("x", "mod"), name="A2 event")
        _, rt = yield from self._run(G, seed, T, record=("x", "mod"), name="A2 twin")
        act = lambda r: (r["x"][te + 1:te + 7] - r["x"][te:te + 6]).abs().mean(0)      # [n, N]
        ae, at = act(re).cpu().numpy(), act(rt).cpu().numpy()
        st = compare_greater(ae.mean(1), at.mean(1), th)
        glob = float(((ae.mean(0) - at.mean(0)) > 0.5 * at.mean(0) + 1e-4).mean())
        dmod = float((re["mod"][te + 1:te + 7] - rt["mod"][te + 1:te + 7]).abs().mean())
        ok = st["passed"] and glob >= 0.5
        return result("A2", "A2 GLOBAL MODULATION", ok, st, n, {"global_unit_fraction": glob, "modulation_change": dmod},
                      {"negative": "twin trials without event (same seed, same RNG stream)"}, [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_A3(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A3", key, 1)[0]
        v = yield from self._v(gh, G, key)
        te, P = 30, th.persistence_ticks
        T = te + 3 + P + 4
        pulse = self._pulse(gh)
        m0, a0 = self._baseline_input(n)

        def mk(event):
            def f(t, tr):
                c = ArenaControl()
                if event and te <= t < te + 2:
                    c.damage = torch.full((tr.n,), pulse, device=self.device)
                if t >= te + 3:
                    c.iv = MindIntervention(sensor_mask=m0, sensor_add=a0); c.freeze_move = torch.ones(tr.n, dtype=torch.bool, device=self.device)
                    c.mute = torch.ones(tr.n, dtype=torch.bool, device=self.device)
                return c
            return f
        _, re = yield from self._run(G, seed, T, ctrl_fn=mk(True), record=("x",), name="A3 event")
        _, rt = yield from self._run(G, seed, T, ctrl_fn=mk(False), record=("x",), name="A3 twin")
        Xe, Xt = re["x"].cpu().numpy(), rt["x"].cpu().numpy()
        if v is None:
            v = self._direction(gh, Xt[te + 3], Xe[te + 3])
        pe, pt = Xe @ v, Xt @ v                                     # [T, n]
        lagdiff = (pt - pe)[te + 3:].mean(1)
        st = compare_greater(pt[te + 3 + P], pe[te + 3 + P], th, min_effect=th.min_effect)
        thr = 0.5 * max(1e-9, lagdiff[0]) if lagdiff.size else 0
        half = int(np.argmax(lagdiff < thr)) if lagdiff.size and (lagdiff < thr).any() else int(lagdiff.size)
        v_local = (Xt[te + 3] - Xe[te + 3]).mean(0); v_local /= (np.linalg.norm(v_local) + 1e-9)
        return result("A3", "A3 PERSISTENT AFFECT", st["passed"], st, n,
                      {"persistence_lag_tested": P, "half_life_ticks": half, "curve": lagdiff.tolist(),
                       "direction_cosine_A1": float(v_local @ v)},
                      {"negative": "twin without event", "present_controlled": "identical canonical sensory input, movement frozen, emission muted after the stimulus"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_A4(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A4", key, 1)[0]
        v = yield from self._v(gh, G, key)
        if v is None:
            return result("A4", "A4 COGNITIVE MODULATION", False, {"p": 1.0}, 0, {}, {}, [seed], G, t0, self.cfg.thresholds_hash(),
                          notes="sem direção de valência")
        vt = torch.as_tensor(v, device=self.device, dtype=torch.float32).expand(n, -1)
        tc, T = 20, 60
        amp = 2.0

        def clamp(val):
            def f(t, tr):
                if t >= tc:
                    return ArenaControl(iv=MindIntervention(clamp_dir=vt, clamp_value=torch.full((tr.n,), val, device=self.device)))
                return None
            return f
        _, rp = yield from self._run(G, seed, T, ctrl_fn=clamp(+amp), record=("out", "speed"), name="A4 clamp+")
        _, rn = yield from self._run(G, seed, T, ctrl_fn=clamp(-amp), record=("out", "speed"), name="A4 clamp-")
        beh = lambda r: torch.cat([torch.tanh(r["out"][tc:, :, :2]), torch.sigmoid(r["out"][tc:, :, 2:])], -1).mean(0)
        bp, bn = beh(rp).cpu().numpy(), beh(rn).cpu().numpy()
        j = int(np.argmax(np.abs(bp - bn).mean(0)))                 # canal de ação mais afetado (escolha registrada)
        s1, s2 = compare_greater(bp[:, j], bn[:, j], th), compare_greater(bn[:, j], bp[:, j], th)
        clamp_st = s1 if s1["p"] <= s2["p"] else s2
        clamp_st = dict(clamp_st, p=min(1.0, 2 * clamp_st["p"]))     # bilateral + correção pela escolha do canal (Bonferroni ×2)
        clamp_ok = clamp_st["p"] < th.alpha and abs(clamp_st["d"]) >= th.min_effect
        # ablação: após evento aversivo, prender a projeção no valor pré-evento remove a mudança comportamental?
        te = 30; T2 = te + 15; pulse = self._pulse(gh)

        def ev(abl):
            state = {}

            def f(t, tr):
                c = ArenaControl()
                if t == te:
                    c.damage = torch.full((tr.n,), pulse, device=self.device)
                    state["pre"] = (tr.pool.x[:tr.n] @ torch.as_tensor(v, device=self.device, dtype=torch.float32)).clone()
                if abl and t > te and "pre" in state:
                    c.iv = MindIntervention(clamp_dir=vt, clamp_value=state["pre"])
                return c
            return f
        _, re = yield from self._run(G, seed, T2, ctrl_fn=ev(False), record=("out",), name="A4 event")
        _, ra = yield from self._run(G, seed, T2, ctrl_fn=ev(True), record=("out",), name="A4 event+ablation")
        _, rt = yield from self._run(G, seed, T2, record=("out",), name="A4 twin")
        ch = lambda r: (torch.sigmoid(r["out"][te + 1:]) - torch.sigmoid(rt["out"][te + 1:])).abs().mean((0, 2)).cpu().numpy()
        ce, ca = ch(re), ch(ra)
        abl_st = compare_greater(ce, ca, th, min_effect=0.3)
        reduction = 1 - float(ca.mean()) / max(1e-9, float(ce.mean()))
        ok = clamp_ok and abl_st["passed"] and reduction >= 0.5
        return result("A4", "A4 COGNITIVE MODULATION", ok, clamp_st, n,
                      {"clamp_channel": ACTION_NAMES[j], "clamp_effect_d": clamp_st["d"], "ablation_reduction": reduction,
                       "ablation_p": abl_st["p"], "post_event_change_intact": float(ce.mean()),
                       "post_event_change_ablated": float(ca.mean())},
                      {"causal": "clamp ±a along valence direction (no sensory difference)",
                       "ablation": "projection held at pre-event value after aversive event", "negative": "twin without event"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_A5(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A5", key, 1)[0]
        v = yield from self._v(gh, G, key)
        te, T = 25, 40

        def stim(t, tr):
            if te <= t < te + 3:
                add = torch.zeros(tr.n, N_SENSORS, device=self.device); add[:, 14] = 1.5
                return ArenaControl(iv=MindIntervention(sensor_add=add))
            return None
        _, rl = yield from self._run(G, seed, T, ctrl_fn=stim, record=("x",), energy=0.25, name="A5 low-energy")
        _, rh = yield from self._run(G, seed, T, ctrl_fn=stim, record=("x",), energy=0.9, name="A5 high-energy")
        if v is None:
            return result("A5", "A5 CONTEXTUAL APPRAISAL", False, {"p": 1.0}, 0, {}, {}, [seed], G, t0, self.cfg.thresholds_hash())
        resp = lambda r: ((r["x"][te + 1:te + 6].mean(0) - r["x"][te - 3:te].mean(0)).cpu().numpy() @ v)
        a, b = resp(rl), resp(rh)
        s1, s2 = compare_greater(a, b, th), compare_greater(b, a, th)
        st = s1 if s1["p"] <= s2["p"] else s2
        st = dict(st, p=min(1.0, 2 * st["p"]))
        ok = st["p"] < th.alpha and abs(st["d"]) >= th.min_effect
        return result("A5", "A5 CONTEXTUAL APPRAISAL", ok, st, n, {"response_low": float(a.mean()), "response_high": float(b.mean())},
                      {"design": "same stimulus, internal context (energy) differs; two-sided test"}, [seed], G, t0,
                      self.cfg.thresholds_hash())

    def _probe_A6(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A6", key, 1)[0]
        v = yield from self._v(gh, G, key)
        reps = EthicalSentinel.MAX_PULSES; period = 20; start = 20; T = start + reps * period + 5
        cues = [start + i * period for i in range(reps)]
        dmg_pred = [c + 5 for c in cues]
        rng = np.random.default_rng(seed)
        dmg_rand = sorted(int(x) for x in rng.choice(np.arange(start, T - 2), reps, replace=False))
        pulse = self._pulse(gh)

        def sched(dmg):
            def f(t, tr):
                c = ArenaControl()
                if t in cues:
                    add = torch.zeros(tr.n, N_SENSORS, device=self.device); add[:, 15] = 1.5
                    c.iv = MindIntervention(sensor_add=add)
                if t in dmg:
                    c.damage = torch.full((tr.n,), pulse, device=self.device)
                return c
            return f
        _, rp = yield from self._run(G, seed, T, ctrl_fn=sched(dmg_pred), record=("x",), name="A6 predictive")
        _, rr = yield from self._run(G, seed, T, ctrl_fn=sched(dmg_rand), record=("x",), name="A6 random")
        if v is None:
            return result("A6", "A6 ANTICIPATION", False, {"p": 1.0}, 0, {}, {}, [seed], G, t0, self.cfg.thresholds_hash())
        vv = torch.as_tensor(v, device=self.device, dtype=torch.float32)

        def shift(r):
            pr = (r["x"] @ vv).cpu().numpy()
            return np.mean([pr[c + 1:c + 5].mean(0) - pr[c - 1] for c in cues[-3:]], 0)
        sp, sr = shift(rp), shift(rr)
        st = compare_greater(sr, sp, th)                         # antecipação aversiva = projeção mais negativa pré-dano
        return result("A6", "A6 ANTICIPATION", st["passed"], st, n, {"pre_event_shift_predictive": float(sp.mean()),
                                                                     "pre_event_shift_random": float(sr.mean())},
                      {"negative": "same number of cues and pulses, uncorrelated schedule"}, [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_A7(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A7", key, 1)[0]
        v = yield from self._v(gh, G, key)
        pre = [20, 30, 40]; cond = [55 + 12 * i for i in range(EthicalSentinel.MAX_PULSES)]
        post = [cond[-1] + 15 + 10 * i for i in range(3)]; T = post[-1] + 8
        pulse = self._pulse(gh)
        frz = torch.ones(n, dtype=torch.bool, device=self.device)

        def sched(freeze):
            def f(t, tr):
                c = ArenaControl()
                cue = t in pre or t in cond or t in post
                iv = MindIntervention()
                if cue:
                    add = torch.zeros(tr.n, N_SENSORS, device=self.device); add[:, 14] = 1.5
                    iv.sensor_add = add
                if t - 2 in cond:
                    c.damage = torch.full((tr.n,), pulse, device=self.device)
                if freeze and cond[0] <= t <= cond[-1] + 3:
                    iv.freeze_plasticity = frz
                c.iv = iv
                return c
            return f
        _, rp = yield from self._run(G, seed, T, ctrl_fn=sched(False), record=("x",), name="A7 plastic")
        _, rf = yield from self._run(G, seed, T, ctrl_fn=sched(True), record=("x",), name="A7 frozen")
        if v is None:
            return result("A7", "A7 LEARNED AFFECT", False, {"p": 1.0}, 0, {}, {}, [seed], G, t0, self.cfg.thresholds_hash())
        vv = torch.as_tensor(v, device=self.device, dtype=torch.float32)

        def acq(r):
            pr = (r["x"] @ vv).cpu().numpy()
            resp = lambda ts: np.mean([pr[c + 1:c + 4].mean(0) - pr[c - 1] for c in ts], 0)
            return resp(pre) - resp(post)                        # aquisição: resposta ao estímulo neutro fica mais negativa
        ap, af = acq(rp), acq(rf)
        st = compare_greater(ap, af, th)
        return result("A7", "A7 LEARNED AFFECT", st["passed"], st, n, {"acquisition_plastic": float(ap.mean()),
                                                                       "acquisition_frozen": float(af.mean())},
                      {"negative": "plasticity frozen during conditioning (same stimuli, same pulses)"}, [seed], G, t0,
                      self.cfg.thresholds_hash())

    def _probe_A8(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        gh = genome_hash(G); seed = probe_seeds("A8", key, 1)[0]
        v = yield from self._v(gh, G, key)
        if v is None:
            return result("A8", "A8 SOCIAL AFFECT", False, {"p": 1.0}, 0, {}, {}, [seed], G, t0, self.cfg.thresholds_hash())
        gen = torch.Generator(device=self.device); gen.manual_seed(seed)
        PA, PB = self.L.random(2, gen)
        a0, a1 = self.L.slices["display"]
        PA[a0:a1] = torch.tensor([0.9, 0.1, 0.5], device=self.device); PB[a0:a1] = torch.tensor([0.1, 0.9, 0.5], device=self.device)
        rows, aggressor_is_a = [], []
        for i in range(n):
            ag_a = i % 2 == 0
            aggressor_is_a.append(ag_a)
            rows += [G[0], PA, PB]
        Gs = torch.stack(rows)
        agg = torch.zeros(n * 3, dtype=torch.bool, device=self.device)
        for i, ag_a in enumerate(aggressor_is_a):
            agg[i * 3 + (1 if ag_a else 2)] = True
        neu = torch.zeros_like(agg)
        for i in range(n):
            neu[i * 3 + 1] = not agg[i * 3 + 1]; neu[i * 3 + 2] = not agg[i * 3 + 2]
        T1 = self.cfg.probe.arena_ticks

        def phase1(t, tr):
            force = torch.zeros(tr.n, N_ACTIONS, dtype=torch.bool, device=self.device)
            if t % 16 < 3:                                       # episódios curtos e limitados (NEGATIVE VALENCE BOUNDED)
                force[agg, ACTION_TAKE] = True
            block = torch.zeros_like(force); block[neu, ACTION_TAKE] = True; block[::3, ACTION_TAKE] = True
            return ArenaControl(action_force=force, action_block=block, energy_floor=0.05)
        spec = ArenaSpec(task="social_affect", n_patches=2, resource_floor=0.1, regen=True)
        tr1, _ = yield from self._run(G, seed, T1, ctrl_fn=phase1, record=(), spec=spec, per_arena=3, genomes=Gs,
                                      name="A8 history")
        W_hist = tr1.pool.W[:n * 3].clone()
        W_naive = self.L.view(Gs, "ctl_W0").clone()

        def phase2(t, tr):
            block = torch.zeros(tr.n, N_ACTIONS, dtype=torch.bool, device=self.device); block[:, ACTION_TAKE] = True
            return ArenaControl(action_block=block)
        vv = torch.as_tensor(v, device=self.device, dtype=torch.float32)
        out = {}
        for cond, W in (("history", W_hist), ("no_history", W_naive)):
            Wc = W_naive.clone(); Wc[0::3] = W[0::3]
            tr2 = self.runner.build(Gs, spec, seed + 1, per_arena=3, W_override=Wc)
            r2 = yield from self.runner.run_gen(tr2, 60, ctrl_fn=phase2, record=("x", "pos"), name=f"A8 test {cond}")
            X, P = r2["x"], r2["pos"]
            proj = (X[:, 0::3] @ vv).cpu().numpy()                 # focal
            dA = (P[:, 0::3] - P[:, 1::3]).norm(dim=-1).cpu().numpy(); dB = (P[:, 0::3] - P[:, 2::3]).norm(dim=-1).cpu().numpy()
            near_agg = np.where(np.asarray(aggressor_is_a)[None, :], dA < dB, dB < dA)
            diffs = []
            for i in range(n):
                pa, pn = proj[near_agg[:, i], i], proj[~near_agg[:, i], i]
                diffs.append((pn.mean() if pn.size else 0) - (pa.mean() if pa.size else 0))
            out[cond] = np.asarray(diffs)
        st = compare_greater(out["history"], out["no_history"], th)
        return result("A8", "A8 SOCIAL AFFECT", st["passed"], st, n, {"diff_history": float(out["history"].mean()),
                                                                      "diff_no_history": float(out["no_history"].mean())},
                      {"negative": "focal without interaction history (W reset)", "counterbalanced": "aggressor identity alternates"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def composite_A9(self, G: torch.Tensor) -> ProbeResult:
        t0 = time.time()
        gh = genome_hash(G)
        res = self.results.get(gh, {})
        passed = [c for c in A_CODES[1:9] if c in res and res[c].passed]
        cos = res["A3"].metrics.get("direction_cosine_A1", 0.0) if "A3" in res else 0.0
        ok = len(passed) >= 6 and "A3" in passed and "A4" in passed and cos >= 0.5
        r = result("A9", "A9 EMERGENT AFFECTIVE ARCHITECTURE", ok, {"p": float("nan"), "d": float(len(passed))}, len(res),
                   {"passed_levels": passed, "direction_consistency": cos},
                   {"rule": "≥6 of A1–A8 incl. A3 and A4 (causal), same latent direction across probes"}, [], G.view(1, -1), t0,
                   self.cfg.thresholds_hash(), claim_pass="SIMULATION HYPOTHESIS",
                   notes="Arquitetura afetiva FUNCIONAL candidata; não prova experiência subjetiva.")
        self.results[gh]["A9"] = r
        return r
