# %% [markdown]
# ## 14 — VALIDATION ENGINE (congelado): escada H0–H12, consistência multiescala, controles
# A evolução **não** altera thresholds, controles, testes, estatística ou regras de evidência: o hash congelado de
# (versão + thresholds + protocolo) é conferido antes de cada probe. Sementes de **holdout** (espaço separado do usado
# pelas ilhas QD). Cada nível tem controle negativo (e positivo quando possível, nos testes de boot).
#
# | nível | pergunta | controle |
# |---|---|---|
# | H0 desenvolvimento organizado | corpo não-trivial, canalizado e diferenciado | genomas aleatórios |
# | H1 corpo funcional | atuação produz locomoção | mesmo corpo sem inervação |
# | H2 controle sensorimotor | sentir recurso melhora ingestão | sensores de recurso mascarados |
# | H3 aprendizado na vida | melhora fim × início da vida | plasticidade congelada |
# | H4 memória | pista passada decodificável com presente controlado | variante sem memória |
# | H5 generalização | desempenho em mundo nunca visto | genoma aleatório no mesmo mundo |
# | H6 modelo preditivo | estado interno prevê sensor futuro além do histórico sensorial | estado embaralhado entre arenas |
# | H7 problema multietapas | pegar → aterrar → cruzar → comer | ação de depositar ablada |
# | H8 aprendizado social | demonstrador treinado × demonstrador ingênuo | demonstrador sem treino |
# | H9 comunicação emergente | emissor informado melhora receptor cego | emissor mudo / canais trocados |
# | H10 transmissão cultural | informação adquirida socialmente persiste | geração não exposta |
# | H11 cultura cumulativa | cadeia de gerações melhora mais que gerações isoladas | cadeias independentes |
# | H12 adaptação aberta | linhagem se adapta a ambientes novos gerados | clones sem variação |

# %%
H_LEVELS = ["H0 ORGANIZED DEVELOPMENT", "H1 FUNCTIONAL BODY", "H2 SENSORIMOTOR CONTROL", "H3 WITHIN-LIFETIME LEARNING",
            "H4 MEMORY", "H5 GENERALIZATION", "H6 PREDICTIVE MODEL", "H7 MULTISTEP PROBLEM SOLVING",
            "H8 SOCIAL LEARNING", "H9 EMERGENT COMMUNICATION", "H10 CULTURAL TRANSMISSION", "H11 CUMULATIVE CULTURE",
            "H12 OPEN-ENDED ADAPTATION"]
H_CODES = [l.split()[0] for l in H_LEVELS]
LEVEL_NAME = {**{l.split()[0]: l for l in H_LEVELS}, **{l.split()[0]: l for l in A_LEVELS}}


class ValidationEngine:
    def __init__(self, cfg: CivConfig, layout: GenomeLayout, runner: ArenaRunner, affect: AffectEngine,
                 device: torch.device):
        self.cfg, self.L, self.runner, self.affect, self.device = cfg, layout, runner, affect, device
        self.dev, self.body = runner.dev, runner.body
        self.frozen = self._protocol_hash()
        self.runs = 0
        self.seconds = 0.0

    def _protocol_hash(self) -> str:
        return hash_obj({"version": VALIDATION_ENGINE_VERSION, "thresholds": asdict(self.cfg.thresholds),
                         "probe": asdict(self.cfg.probe), "levels": H_LEVELS + A_LEVELS})[:16]

    def verify_frozen(self):
        h = self._protocol_hash()
        if h != self.frozen:
            raise RuntimeError(f"Validation Engine alterado em execução (hash {h} ≠ congelado {self.frozen}) — bloqueado")

    def run_gen(self, code: str, G: torch.Tensor, key: str = "") -> Generator:
        self.verify_frozen()
        t0 = time.time()
        G = G.view(1, -1).to(self.device)
        key = key or genome_hash(G)
        if code.startswith("A"):
            res = (self.affect.composite_A9(G) if code == "A9" else (yield from self.affect.probe_gen(code, G, key)))
        elif code == "MS":
            res = yield from self._probe_MS(G, key)
        else:
            res = yield from getattr(self, f"_probe_{code}")(G, key)
        self.runs += 1
        self.seconds += time.time() - t0
        return res

    # ------------------------------------------------------------------ utilidades
    def _spec_forage(self) -> ArenaSpec:
        return ArenaSpec(task="forage", n_patches=3, resource_floor=0.05)

    def _run(self, Gs, spec, seed, ticks, ctrl_fn=None, record=("gain",), per_arena=1, W=None, x=None, name="",
             bodies_fn=None, energy=0.8):
        tr = self.runner.build(Gs, spec, seed, per_arena=per_arena, W_override=W, x_override=x, energy=energy)
        bodies = bodies_fn(tr) if bodies_fn else None
        rec = yield from self.runner.run_gen(tr, ticks, ctrl_fn=ctrl_fn, record=record, name=name, bodies=bodies)
        return tr, rec

    def _rep(self, G, n):
        return G.expand(n, -1).contiguous()

    # ------------------------------------------------------------------ H0–H1 (corpo)
    def _probe_H0(self, G, key):
        t0, th = time.time(), self.cfg.thresholds
        k = max(6, 2 * self.cfg.probe.dev_noise_samples)
        seeds = probe_seeds("H0", key, k)
        sd = torch.tensor([s % 100003 for s in seeds], device=self.device)
        bp = self.dev.develop(self._rep(G, k), sd)
        occ = bp.occ.float()
        nodes = occ.sum(1)
        iou = (occ @ occ.T) / ((occ.unsqueeze(1) + occ.unsqueeze(0)).clamp(max=1).sum(-1) + 1e-9)
        canal = ((iou.sum(1) - 1) / max(1, k - 1))

        def differentiation(b):
            m = b.occ.unsqueeze(-1).float()
            mu = (b.props * m).sum(1) / m.sum(1).clamp(min=1)
            var = (((b.props - mu.unsqueeze(1)) ** 2) * m).sum(1) / m.sum(1).clamp(min=1)
            return var.sqrt().mean(1)
        diff_c = differentiation(bp)
        org = (canal * diff_c).cpu().numpy()
        gen = torch.Generator(device=self.device); gen.manual_seed(derive_seed("H0ctrl", key))
        Rg = self.L.random(16, gen)
        ctrl = []
        for j in range(3):
            ctrl.append(self.dev.develop(Rg, torch.full((16,), 101 + j, device=self.device)))
        co = torch.stack([c.occ.float() for c in ctrl])                           # [3,16,N]
        ciou = []
        for a in range(3):
            for b in range(a + 1, 3):
                ciou.append((co[a] * co[b]).sum(-1) / ((co[a] + co[b]).clamp(max=1).sum(-1) + 1e-9))
        ccanal = torch.stack(ciou).mean(0)
        corg = (ccanal * differentiation(ctrl[0])).cpu().numpy()
        st = compare_greater(org, corg, th)
        N = self.dev.N
        nontrivial = float(((nodes >= th.dev_min_nodes) & (nodes <= 0.9 * N)).float().mean())
        ok = st["passed"] and nontrivial >= 0.8 and float(canal.mean()) >= th.dev_canalization
        sym = self.dev.symmetry(bp.occ)
        yield
        return result("H0", "H0 ORGANIZED DEVELOPMENT", ok, st, k,
                      {"nodes_mean": float(nodes.mean()), "nontrivial_fraction": nontrivial, "canalization": float(canal.mean()),
                       "differentiation": float(diff_c.mean()), "bilateral_x": float(sym["bilateral_x"].mean()),
                       "radial_z": float(sym["radial_z"].mean())},
                      {"negative": "16 random genomes (organization = canalization × property differentiation)"},
                      seeds, G, t0, self.cfg.thresholds_hash())

    def _probe_H1(self, G, key):
        t0, th = time.time(), self.cfg.thresholds
        k = 8
        seeds = probe_seeds("H1", key, k)
        sd = torch.tensor([s % 100003 for s in seeds], device=self.device)
        cap = self.body.measure_capability(self._rep(G, k), sd)
        yield
        capp = self.body.measure_capability(passive_body(self.L, self._rep(G, k)), sd)
        yield
        st = compare_greater(cap[:, 0].cpu().numpy(), capp[:, 0].cpu().numpy(), th)
        # robustez funcional a dano: remove ~25% dos nós (lado +x) e mede a marcha restante
        bp = self.dev.develop(self._rep(G, k), sd)
        gx = self.dev.grid[:, 0].float()
        cut = torch.quantile(gx[bp.occ[0]], 0.75) if bool(bp.occ[0].any()) else gx.max()
        dmg = bp.occ & (gx.unsqueeze(0) >= cut) & (bp.occ.sum(1, keepdim=True) > 2)
        bp.occ = bp.occ & ~dmg
        capd = self.body.measure_blueprint(bp)
        retained = float((capd[:, 0].mean() / cap[:, 0].mean().clamp(min=1e-9)).clamp(max=5))
        ok = st["passed"] and st["mean_a"] >= th.body_min_speed
        return result("H1", "H1 FUNCTIONAL BODY", ok, st, k, {"speed": st["mean_a"], "passive_speed": st["mean_b"],
                                                             "cost": float(cap[:, 1].mean()), "nodes": float(cap[:, 2].mean()),
                                                             "damage_speed_retained": retained},
                      {"negative": "same developed body, motor innervation removed (passive)",
                       "limitation": "no structural regrowth after damage is modeled (functional robustness only)"},
                      seeds, G, t0, self.cfg.thresholds_hash())

    # ------------------------------------------------------------------ H2–H7 (mente individual)
    def _probe_H2(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H2", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        tr, ri = yield from self._run(self._rep(G, n), self._spec_forage(), seed, T, record=("gain", "pos"), name="H2 intact")
        mask = sensor_mask(n, SENSOR_RESOURCE, self.device)
        _, rm = yield from self._run(self._rep(G, n), self._spec_forage(), seed, T, name="H2 masked",
                                     ctrl_fn=lambda t, tr_: ArenaControl(iv=MindIntervention(sensor_mask=mask)))
        a, b = ri["gain"].sum(0).cpu().numpy(), rm["gain"].sum(0).cpu().numpy()
        st = compare_greater(a, b, th)
        gain = (st["mean_a"] - st["mean_b"]) / max(1e-6, abs(st["mean_b"]) + 1e-3)
        ok = st["passed"] and gain >= th.sensorimotor_gain
        return result("H2", "H2 SENSORIMOTOR CONTROL", ok, st, n, {"intake_intact": st["mean_a"], "intake_masked": st["mean_b"],
                                                                  "relative_gain": gain},
                      {"negative": "resource sensors masked, same layouts/seeds"}, [seed], G, t0, self.cfg.thresholds_hash(),
                      replay=make_replay(tr, ri))

    def _probe_H3(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H3", key, 1)[0]
        T = 2 * self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="learn", n_patches=2, regen=True, resource_floor=0.0)
        _, rp = yield from self._run(self._rep(G, n), spec, seed, T, name="H3 plastic")
        frz = torch.ones(n, dtype=torch.bool, device=self.device)
        _, rf = yield from self._run(self._rep(G, n), spec, seed, T, name="H3 frozen",
                                     ctrl_fn=lambda t, tr_: ArenaControl(iv=MindIntervention(freeze_plasticity=frz)))
        third = T // 3
        imp = lambda r: (r["gain"][-third:].sum(0) - r["gain"][:third].sum(0)).cpu().numpy()
        early = float(rp["gain"][:third].sum(0).mean())
        a, b = imp(rp), imp(rf)
        st = compare_greater(a, b, th)
        ok = st["passed"] and (st["mean_a"] - st["mean_b"]) >= th.learning_gain * max(1e-3, early)
        return result("H3", "H3 WITHIN-LIFETIME LEARNING", ok, st, n, {"improvement_plastic": st["mean_a"],
                                                                      "improvement_frozen": st["mean_b"], "early_intake": early},
                      {"negative": "plasticity frozen (identical layouts, identical babbling noise stream)"}, [seed], G, t0,
                      self.cfg.thresholds_hash())

    def _memory_decode(self, G, seed, delays, n, name):
        cue_t = 6
        T = cue_t + max(delays) + 2
        lab = torch.arange(n, device=self.device) % 2
        m0 = torch.zeros(n, N_SENSORS, device=self.device)
        a0 = torch.zeros(n, N_SENSORS, device=self.device); a0[:, 0] = 1; a0[:, 23] = 0.6; a0[:, 24] = 0.9; a0[:, 4] = 1; a0[:, 5] = 0.6
        frz = torch.ones(n, dtype=torch.bool, device=self.device)

        def ctrl(t, tr):
            if t < cue_t:
                add = a0.clone(); add[:, 12] += (lab == 0).float() * 1.5; add[:, 13] += (lab == 1).float() * 1.5
                return ArenaControl(iv=MindIntervention(sensor_mask=m0, sensor_add=add), freeze_move=frz, mute=frz)
            return ArenaControl(iv=MindIntervention(sensor_mask=m0, sensor_add=a0), freeze_move=frz, mute=frz)
        spec = ArenaSpec(task="memory", n_patches=0, resource_floor=0.0)
        _, r = yield from self._run(G, spec, seed, T, ctrl_fn=ctrl, record=("x",), name=name)
        X = r["x"].cpu().numpy(); y = lab.cpu().numpy()
        out = {}
        for d in delays:
            out[d] = decode_report(X[cue_t + d], y, 2, self.cfg.probe.permutations, self.cfg.probe.folds, seed + d, nonlinear=False)
        return out

    def _probe_H4(self, G, key):
        t0, th = time.time(), self.cfg.thresholds
        n = 4 * self.cfg.probe.trials
        seed = probe_seeds("H4", key, 1)[0]
        delays = list(self.cfg.probe.memory_delays)
        dc = yield from self._memory_decode(self._rep(G, n), seed, delays, n, "H4 memory")
        dn = yield from self._memory_decode(memoryless(self.L, self._rep(G, n)), seed, delays, n, "H4 memoryless control")
        dmax = max(delays)
        acc, p = dc[dmax]["linear_acc"], dc[dmax]["linear_p"]
        nacc = norm_acc(acc, 0.5)
        ctrl_fail = not (dn[dmax]["linear_p"] < th.alpha and norm_acc(dn[dmax]["linear_acc"], 0.5) >= th.min_norm_acc)
        ok = p < th.alpha and nacc >= th.min_norm_acc and ctrl_fail
        st = {"p": p, "d": dc[dmax].get("effect_size", 0.0), "ci": (float("nan"), float("nan"))}
        return result("H4", "H4 MEMORY", ok, st, n, {"acc_by_delay": {str(d): dc[d]["linear_acc"] for d in delays},
                                                     "p_by_delay": {str(d): dc[d]["linear_p"] for d in delays},
                                                     "control_acc_by_delay": {str(d): dn[d]["linear_acc"] for d in delays},
                                                     "normalized_acc_longest": nacc},
                      {"present_controlled": "identical canonical input after cue, movement frozen, emission muted",
                       "negative": "memoryless variant (no recurrence, τ=1, no plasticity) must fail", "control_failed": ctrl_fail},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_H5(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        s_train = probe_seeds("H5", key, 1, holdout=False)[0]
        s_hold = probe_seeds("H5", key, 1, holdout=True)[0]
        gen = torch.Generator(device=self.device); gen.manual_seed(derive_seed("H5rand", key))
        R = self.L.random(n, gen)
        train = ArenaSpec(task="forage", n_patches=3, resource_floor=0.1)
        novel = ArenaSpec(task="novel", n_patches=6, patch_radius=0, cold=0.3, storm=0.15, resource_floor=0.03)
        T = self.cfg.probe.arena_ticks
        _, rc_tr = yield from self._run(self._rep(G, n), train, s_train, T, name="H5 train-like")
        _, rr_tr = yield from self._run(R, train, s_train, T, name="H5 random train-like")
        _, rc_ho = yield from self._run(self._rep(G, n), novel, s_hold, T, name="H5 holdout")
        _, rr_ho = yield from self._run(R, novel, s_hold, T, name="H5 random holdout")
        tot = lambda r: r["gain"].sum(0).cpu().numpy()
        a, b = tot(rc_ho), tot(rr_ho)
        st = compare_greater(a, b, th)
        norm = lambda c, r: (np.mean(c) - np.mean(r)) / (abs(np.mean(r)) + 1e-3)
        ratio = norm(a, b) / max(1e-6, norm(tot(rc_tr), tot(rr_tr))) if norm(tot(rc_tr), tot(rr_tr)) > 0 else 0.0
        ok = st["passed"] and ratio >= 0.5
        return result("H5", "H5 GENERALIZATION", ok, st, n, {"holdout_intake": st["mean_a"], "holdout_random": st["mean_b"],
                                                            "generalization_ratio": ratio, "holdout_world": asdict(novel)},
                      {"negative": "random genomes in the same never-seen world", "holdout": "seed space never used by QD",
                       "holdout_surprise_rule": "ratio < 0.5 blocks promotion"}, [s_train, s_hold], G, t0, self.cfg.thresholds_hash())

    def _probe_H6(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H6", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        _, r = yield from self._run(self._rep(G, n), self._spec_forage(), seed, T, record=("x", "sense_cache"), name="H6 predict")
        _, rm = yield from self._run(memoryless(self.L, self._rep(G, n)), self._spec_forage(), seed, T,
                                     record=("x", "sense_cache"), name="H6 memoryless")
        k = 3
        ch = [1, 2, 3, 23]

        def gain_of(rec, perm=None):
            X = rec["x"].cpu().numpy(); S = rec["sense_cache"].cpu().numpy()
            ts = np.arange(2, T - k)
            if perm is not None:
                X = X[:, perm]
            base = np.concatenate([S[ts], S[ts - 1], S[ts - 2]], -1)          # [t, n, 3I]
            full = np.concatenate([base, X[ts]], -1)
            Y = S[ts + k][..., ch]
            g = np.broadcast_to(np.arange(n)[None, :], (len(ts), n))
            fl = lambda A: A.reshape(-1, A.shape[-1])
            r2b = ridge_r2_cv(fl(base), fl(Y), g.reshape(-1), lam=5.0)
            r2f = ridge_r2_cv(fl(full), fl(Y), g.reshape(-1), lam=5.0)
            return r2f - r2b, r2f, r2b
        gobs, r2f, r2b = gain_of(r)
        rng = np.random.default_rng(seed)
        null = [gain_of(r, rng.permutation(n))[0] for _ in range(max(10, self.cfg.probe.permutations // 3))]
        p = (sum(1 for v in null if v >= gobs) + 1) / (len(null) + 1)
        gm = gain_of(rm)[0]
        ok = gobs >= th.prediction_gain and p < th.alpha
        st = {"p": p, "d": (gobs - float(np.mean(null))) / (float(np.std(null)) + 1e-9), "ci": (float("nan"), float("nan"))}
        return result("H6", "H6 PREDICTIVE MODEL", ok, st, n, {"r2_gain": gobs, "r2_full": r2f, "r2_sensor_history": r2b,
                                                              "memoryless_gain": gm, "horizon": k},
                      {"baseline": "current + 2 past sensor frames", "negative": "internal state shuffled across arenas",
                       "memoryless": "reported"}, [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_H7(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H7", key, 1)[0]
        T = 2 * self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="barrier", water_barrier=True, deposit=3.0, spawn_side="left", n_patches=3, resource_floor=0.0)
        tr, ri = yield from self._run(self._rep(G, n), spec, seed, T, record=("gain", "pos"), name="H7 intact")
        crossed_i = ArenaRunner.crossed(tr).cpu().numpy(); tech_i = tr.pool.tech[:n].cpu().numpy()
        blk = torch.zeros(n, N_ACTIONS, dtype=torch.bool, device=self.device); blk[:, ACTION_PLACE] = True
        tr2, ra = yield from self._run(self._rep(G, n), spec, seed, T, ctrl_fn=lambda t, x: ArenaControl(action_block=blk),
                                       name="H7 place-ablated")
        crossed_a = ArenaRunner.crossed(tr2).cpu().numpy()
        k1, k2 = int(crossed_i.sum()), int(crossed_a.sum())
        p = fisher_greater(k1, n, k2, n)
        seq = int(((tech_i & 1) > 0)[crossed_i].sum()) if crossed_i.any() else 0
        ok = p < th.alpha and k1 / n >= 0.3 and seq >= 1
        st = {"p": p, "d": (k1 - k2) / n, "ci": (float("nan"), float("nan"))}
        return result("H7", "H7 MULTISTEP PROBLEM SOLVING", ok, st, n,
                      {"success_intact": k1 / n, "success_ablated": k2 / n, "crossers_who_filled_water": seq,
                       "tool_test": "material placed in water causally enables reaching resource"},
                      {"negative": "place action ablated (cannot fill water)", "test": "one-sided Fisher exact"},
                      [seed], G, t0, self.cfg.thresholds_hash(), replay=make_replay(tr, ri))

    # ------------------------------------------------------------------ H8–H11 (social / cultura)
    def _train(self, G, n, spec, seed, T, name):
        tr, _ = yield from self._run(self._rep(G, n), spec, seed, T, record=(), name=name)
        return tr.pool.W[:n].clone()

    def _social_phase(self, G, n, spec, seed, T, W_model, W_naive, name, freeze_naive=False):
        """Pares (modelo, aprendiz) por arena; devolve W do aprendiz após a fase."""
        Gs = self._rep(G, 2 * n)
        W = torch.empty(2 * n, *W_model.shape[1:], device=self.device)
        W[0::2], W[1::2] = W_model, W_naive
        ctrl_fn = None
        if freeze_naive:
            fz = torch.zeros(2 * n, dtype=torch.bool, device=self.device); fz[1::2] = True
            ctrl_fn = lambda t, tr: ArenaControl(iv=MindIntervention(freeze_plasticity=fz))
        tr, r = yield from self._run(Gs, spec, seed, T, per_arena=2, W=W, ctrl_fn=ctrl_fn, record=("emit", "speed"), name=name)
        return tr.pool.W[1::2].clone(), tr.pool.signature[1::2].clone(), r

    def _probe_H8(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H8", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="social_learning", n_patches=2, regen=True, resource_floor=0.0)
        W0 = self.L.view(self._rep(G, n), "ctl_W0").clone()
        Wm = yield from self._train(G, n, spec, seed, T, "H8 demonstrator training")
        Wd, _, rd = yield from self._social_phase(G, n, spec, seed, T, Wm, W0, "H8 demonstration")
        Ws, _, rs = yield from self._social_phase(G, n, spec, seed, T, W0, W0, "H8 sham demonstration")
        _, rt = yield from self._run(self._rep(G, n), spec, seed, T, W=Wd, name="H8 naive test (demonstrated)")
        _, rc = yield from self._run(self._rep(G, n), spec, seed, T, W=Ws, name="H8 naive test (sham)")
        a, b = rt["gain"].sum(0).cpu().numpy(), rc["gain"].sum(0).cpu().numpy()
        st = compare_greater(a, b, th)
        teach = float((rd["emit"][:, 0::2].abs().sum(-1).mean() - rs["emit"][:, 0::2].abs().sum(-1).mean()).item())
        return result("H8", "H8 SOCIAL LEARNING", st["passed"], st, n, {"naive_intake_demonstrated": st["mean_a"],
                                                                       "naive_intake_sham": st["mean_b"],
                                                                       "teaching_candidate_emission_delta": teach},
                      {"negative": "sham demonstrator (same genome, untrained weights)",
                       "isolation": "test phase alone, fresh resources, transient state reset (only learned weights carried)"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_H9(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H9", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="signal", n_patches=2, resource_floor=0.0)
        Gs = self._rep(G, 2 * n)
        vis = torch.zeros(2 * n, dtype=torch.bool, device=self.device); vis[0::2] = True
        mute = vis.clone()
        recv = ~vis
        perm = torch.tensor([1, 0, 3, 2], device=self.device)

        def mk(kind):
            def f(t, tr):
                c = ArenaControl(resource_visible=vis)
                if kind == "muted":
                    c.mute = mute
                if kind == "swapped":
                    c.sound_override = lambda snd: torch.where(recv[:snd.shape[0]].unsqueeze(1), snd[:, perm], snd)
                return c
            return f
        tr, ri = yield from self._run(Gs, spec, seed, T, per_arena=2, ctrl_fn=mk("intact"),
                                      record=("gain", "emit", "sense_cache", "pos"), name="H9 intact")
        _, rm = yield from self._run(Gs, spec, seed, T, per_arena=2, ctrl_fn=mk("muted"), record=("gain",), name="H9 sender muted")
        _, rs = yield from self._run(Gs, spec, seed, T, per_arena=2, ctrl_fn=mk("swapped"), record=("gain",), name="H9 swapped")
        g = lambda r: r["gain"][:, 1::2].sum(0).cpu().numpy()
        st = compare_greater(g(ri), g(rm), th)
        sw = compare_greater(g(ri), g(rs), th)
        E = ri["emit"][:, 0::2].cpu().numpy(); S = ri["sense_cache"][:, 0::2].cpu().numpy()
        loud = E.max(-1) > 0.3
        sig = np.where(loud, E.argmax(-1), 4).reshape(-1)
        world = np.digitize(S[..., 1], [0.05, 0.3]).reshape(-1)
        rng = np.random.default_rng(seed)
        mi, pmi = mi_permutation(sig, world, max(20, self.cfg.probe.permutations), rng)
        ok = st["passed"] and pmi < th.alpha
        return result("H9", "H9 EMERGENT COMMUNICATION", ok, st, n,
                      {"receiver_intake_intact": st["mean_a"], "receiver_intake_muted": st["mean_b"],
                       "receiver_intake_swapped": sw["mean_b"], "swap_p": sw["p"], "mi_signal_world_bits": mi, "mi_p": pmi,
                       "sender_loud_fraction": float(loud.mean())},
                      {"negative": "sender muted (sound channel interrupted)", "swap": "receiver channels permuted",
                       "grounding": "I(sender signal; sender's resource state) with permutation null"},
                      [seed], G, t0, self.cfg.thresholds_hash(), replay=make_replay(tr, ri))

    def _probe_H10(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H10", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="culture", n_patches=2, regen=True, resource_floor=0.0)
        W0 = self.L.view(self._rep(G, n), "ctl_W0").clone()
        Wm = yield from self._train(G, n, spec, seed, T, "H10 model generation")
        tr_m, _ = yield from self._run(self._rep(G, n), spec, seed + 1, 2 * T, W=Wm, record=(), name="H10 model signature")
        sig_m = tr_m.pool.signature[:n].clone()
        We, _, _ = yield from self._social_phase(G, n, spec, seed, T, Wm, W0, "H10 exposure")
        Wu = yield from self._train(G, n, spec, seed + 7, T, "H10 unexposed (learns alone for the same time)")
        tr_e, re_ = yield from self._run(self._rep(G, n), spec, seed + 1, 2 * T, W=We, name="H10 persistence exposed")
        tr_u, ru = yield from self._run(self._rep(G, n), spec, seed + 1, 2 * T, W=Wu, name="H10 persistence unexposed")
        late = lambda r: r["gain"][T:].sum(0).cpu().numpy()
        st = compare_greater(late(re_), late(ru), th)
        de = (tr_e.pool.signature[:n] - sig_m).norm(dim=-1).cpu().numpy()
        du = (tr_u.pool.signature[:n] - sig_m).norm(dim=-1).cpu().numpy()
        sim = compare_greater(du, de, th, min_effect=0.3)
        ok = st["passed"] and sim["p"] < th.alpha
        return result("H10", "H10 CULTURAL TRANSMISSION", ok, st, n,
                      {"late_intake_exposed": st["mean_a"], "late_intake_unexposed": st["mean_b"],
                       "behavior_distance_to_model_exposed": float(de.mean()), "behavior_distance_unexposed": float(du.mean()),
                       "similarity_p": sim["p"]},
                      {"negative": "same-age generation not exposed (learned alone for the same time)",
                       "persistence": "measured in the late half of a 2T solo phase after the model is removed"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_H11(self, G, key):
        t0, th, n = time.time(), self.cfg.thresholds, self.cfg.probe.trials
        seed = probe_seeds("H11", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        spec = ArenaSpec(task="cumulative", n_patches=3, regen=True, resource_floor=0.0)
        W0 = self.L.view(self._rep(G, n), "ctl_W0").clone()
        K = 3
        chain, indep = [], []
        Wc = yield from self._train(G, n, spec, seed, T, "H11 chain gen0")
        Wi = Wc.clone()
        for g in range(1, K + 1):
            Wl, _, _ = yield from self._social_phase(G, n, spec, seed + g, T, Wc, W0, f"H11 chain gen{g} learning")
            trc, rc = yield from self._run(self._rep(G, n), spec, seed + 100 + g, T, W=Wl, name=f"H11 chain gen{g} solo")
            chain.append(rc["gain"].sum(0).cpu().numpy())
            Wc = trc.pool.W[:n].clone()
            Wil = yield from self._train(G, n, spec, seed + 50 + g, T, f"H11 independent gen{g}")
            _, ri = yield from self._run(self._rep(G, n), spec, seed + 100 + g, T, W=Wil, name=f"H11 independent gen{g} solo")
            indep.append(ri["gain"].sum(0).cpu().numpy())
        C, I = np.stack(chain), np.stack(indep)                       # [K, n]
        xs = np.arange(1, K + 1) - (K + 1) / 2
        slope = lambda M: (xs[:, None] * (M - M.mean(0))).sum(0) / (xs ** 2).sum()
        st = compare_greater(slope(C), slope(I), th)
        ok = st["passed"] and float(C[-1].mean()) > float(I[-1].mean())
        return result("H11", "H11 CUMULATIVE CULTURE", ok, st, n, {"chain_by_gen": C.mean(1).tolist(),
                                                                  "independent_by_gen": I.mean(1).tolist()},
                      {"negative": "independent generations (same learning time, no transmission)"},
                      [seed], G, t0, self.cfg.thresholds_hash())

    def _probe_H12(self, G, key):
        """Adaptação aberta (nível de POPULAÇÃO, por seleção — não é aprendizado individual): em ambientes novos gerados
        proceduralmente, linhagens com variação melhoram mais que clones sem variação."""
        t0, th = time.time(), self.cfg.thresholds
        seeds = probe_seeds("H12", key, 3)
        envs = [ArenaSpec(task="gen_cold", n_patches=2, cold=0.7, deposit=2.0, resource_floor=0.05),
                ArenaSpec(task="gen_barrier", water_barrier=True, deposit=3.0, spawn_side="left", n_patches=3, resource_floor=0.03),
                ArenaSpec(task="gen_storm", n_patches=4, storm=0.5, deposit=2.0, resource_floor=0.05)]
        gens, pop = 4, max(8, self.cfg.probe.trials)
        gen = torch.Generator(device=self.device); gen.manual_seed(derive_seed("H12", key))
        imps_var, imps_clone = [], []
        for env, sd in zip(envs, seeds):
            for variant in ("var", "clone"):
                P = self._rep(G, pop).clone()
                best0, bestN = None, None
                for g in range(gens):
                    tr, r = yield from self._run(P, env, sd + g, self.cfg.probe.arena_ticks, record=("gain", "alive"),
                                                 name=f"H12 {env.task} {variant} g{g}")
                    fit = (r["gain"].sum(0) + 0.5 * r["alive"].float().mean(0))
                    if g == 0:
                        best0 = float(fit.mean())
                    bestN = float(fit.mean())
                    order = torch.argsort(fit, descending=True)
                    elite = P[order[: max(2, pop // 4)]]
                    rep = elite[torch.randint(0, elite.shape[0], (pop,), generator=gen, device=self.device)]
                    P = self.L.mutate(rep, gen) if variant == "var" else self._rep(G, pop).clone()
                (imps_var if variant == "var" else imps_clone).append(bestN - best0)
        a, b = np.asarray(imps_var), np.asarray(imps_clone)
        wins = int((a > b).sum())
        st = {"p": float(binom_p_greater(wins, len(a), 0.5)), "d": float((a - b).mean() / (np.std(a - b) + 1e-6)),
              "ci": (float("nan"), float("nan"))}
        ok = wins == len(a) and float((a - b).mean()) > 0
        return result("H12", "H12 OPEN-ENDED ADAPTATION", ok, st, len(a), {"improvement_variation": a.tolist(),
                                                                         "improvement_clones": b.tolist(), "wins": wins},
                      {"negative": "clonal lineage without variation in the same generated environments",
                       "scope": "population-level adaptation by selection (not individual learning)"},
                      seeds, G, t0, self.cfg.thresholds_hash(), claim_pass="SIMULATION HYPOTHESIS",
                      notes="Escopo limitado a 3 ambientes gerados; abertura ilimitada não é demonstrável num teste finito.")

    # ------------------------------------------------------------------ consistência multiescala (F1 × F3)
    def _probe_MS(self, G, key):
        t0, n = time.time(), self.cfg.probe.trials
        seed = probe_seeds("MS", key, 1)[0]
        T = self.cfg.probe.arena_ticks
        spec = self._spec_forage()
        _, r1 = yield from self._run(self._rep(G, n), spec, seed, T, record=("gain", "speed"), name="MS F1")

        def bodies_fn(tr):
            reg = BodyRegistry(self.cfg, self.dev, self.body, tr.n, self.cfg.body_substeps)
            sd = torch.tensor([s % 100003 for s in probe_seeds("MSdev", key, tr.n)], device=self.device)
            bs = reg.attach(torch.arange(tr.n, device=self.device), tr.pool.aid[:tr.n], tr.pool.genome[:tr.n], sd)
            tr.pool.body_slot[:tr.n] = bs.int(); tr.pool.fidelity[:tr.n] = 3
            return {3: reg}
        _, r3 = yield from self._run(self._rep(G, n), spec, seed, T, record=("gain", "speed"), name="MS F3", bodies_fn=bodies_fn)
        d1, d3 = r1["speed"].sum(0).cpu().numpy(), r3["speed"].sum(0).cpu().numpy()
        g1, g3 = r1["gain"].sum(0).cpu().numpy(), r3["gain"].sum(0).cpu().numpy()
        rel = lambda a, b: float(abs(np.mean(a) - np.mean(b)) / (abs(np.mean(a)) + 1e-6))
        err_d, err_g = rel(d1, d3), rel(g1, g3)
        ok = err_d <= 0.5 and err_g <= 0.5
        st = {"p": float("nan"), "d": err_d, "ci": (float("nan"), float("nan"))}
        return result("MS", "MULTISCALE CONSISTENCY F1×F3", ok, st, n,
                      {"path_F1": float(d1.mean()), "path_F3": float(d3.mean()), "rel_error_path": err_d,
                       "intake_F1": float(g1.mean()), "intake_F3": float(g3.mean()), "rel_error_intake": err_g},
                      {"design": "same genome, same layouts; F1 surrogate vs full developmental body physics"},
                      [seed], G, t0, self.cfg.thresholds_hash(), claim_pass="BENCHMARKED")
