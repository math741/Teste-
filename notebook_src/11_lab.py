# %% [markdown]
# ## 11. GENESIS LAB — orquestração do ciclo `CREATE → EVOLVE → OBSERVE → PERTURB → MEASURE → REJECT → PRESERVE → REPLICATE → GENERALIZE → CONTINUE`
# * Universe Evolution Engine (este módulo) e Validation Engine (módulo 6) são separados: a evolução **nunca** altera
#   probes, thresholds, checkpoint ou segurança (core imutável; meta-evolução só toca parâmetros de mutação do genoma).
# * Seleção: MAP-Elites + novelty + frentes de Pareto multiobjetivo + limite por espécie + imigrantes aleatórios.
# * Orçamento: triagem barata para todos; experimentos caros só para candidatos promovidos pelo Scientist.
# * Estagnação: medidas graduais (mutação ↑, pressão de novidade ↑, novas espécies, reinício parcial, mudança ambiental).
#   Elites históricas nunca são apagadas.

# %%
def pareto_ranks(Obj: np.ndarray) -> np.ndarray:
    n = len(Obj)
    ranks = np.zeros(n, int)
    if n == 0:
        return ranks
    dom = (Obj[:, None, :] >= Obj[None, :, :]).all(-1) & (Obj[:, None, :] > Obj[None, :, :]).any(-1)
    remaining = np.ones(n, bool)
    r = 0
    while remaining.any():
        dominated = (dom[remaining][:, remaining]).any(0)
        idx = np.where(remaining)[0]
        front = idx[~dominated]
        if front.size == 0:
            front = idx
        ranks[front] = r
        remaining[front] = False
        r += 1
    return ranks


PROBE_ALIAS = {"channel_noise": "channel", "functional_recovery": "channel"}


class GenesisLab:
    def __init__(self, cfg: GenesisConfig, hw: Dict[str, Any], paths: GenesisPaths, logger: EventLogger,
                 device: torch.device):
        self.cfg, self.hw, self.paths, self.logger, self.device = cfg, hw, paths, logger, device
        self.space = ParamSpace(cfg)
        self.sim = Simulator(cfg, device)
        self.envs = make_environment_suite(cfg.master_seed)
        self.screener = Screener(self.sim, cfg)
        self.curriculum: Dict[str, float] = {"perturbation_mult": 1.0, "memory_delay_mult": 1.0, "channel_distance_add": 0}
        self.ve = ValidationEngine(self.sim, cfg, self.envs, self.screener, self.space, self.curriculum)
        self.ladder = DiscoveryLadder(cfg.thresholds)
        self.scientist = GenesisScientist(cfg)
        self.ckpt = CheckpointManager(paths, logger)
        self.store: Dict[str, Genome] = {}
        self.population: List[str] = []
        self.desc: Dict[str, Dict[str, Any]] = {}
        self.vec: Dict[str, np.ndarray] = {}
        self.quality: Dict[str, float] = {}
        self.objectives: Dict[str, np.ndarray] = {}
        self.novelty_score: Dict[str, float] = {}
        self.results: Dict[str, TestResult] = {}
        self.tests: Dict[str, Dict[str, str]] = {}
        self.levels: Dict[str, int] = {}
        self.evidence: Dict[str, float] = {}
        self.audit: List[Dict[str, Any]] = []
        self.map = MapElites(cfg.map_axes, cfg.map_bins)
        self.novelty = NoveltyArchive(cfg.novelty_k, cfg.novelty_archive_max)
        self.zoo = Zoo()
        self.anomalies = SimpleArchive("anomaly")
        self.failures = SimpleArchive("failure")
        self.champions = SimpleArchive("champion_candidates")
        self.pantheon: List[Dict[str, Any]] = []
        self.lineage = Lineage()
        self.species = Speciation(cfg.species_threshold)
        self.surprise = SurpriseModel(len(DESCRIPTOR_NAMES))
        self.bandit = OperatorBandit(MUTATION_OPERATORS)
        self.discoveries: Dict[str, Dict[str, Any]] = {}
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.counters = {"generation": 0, "evaluations": 0, "sim_steps": 0, "total_runtime": 0.0, "sessions": 0,
                         "experiments": 0, "oom_events": 0, "discovery_events": 0}
        self.history: List[Dict[str, Any]] = []
        self.stagnation = {"stage": 0, "best_quality": -1.0, "best_coverage": 0.0, "best_level": -1, "since": 0,
                           "sigma_mult": 1.0, "novelty_weight": 1.0}
        self.rng = np.random.default_rng(cfg.master_seed)
        self.champion: Optional[Dict[str, Any]] = None
        self.champion_lab_runs = 0
        self.metric_flags: List[Dict[str, Any]] = []
        self.session_start = time.time()
        self.runtime_at_start = 0.0
        self.last_checkpoint_ts = time.time()
        self.last_dashboard_ts = 0.0
        self.best_level_global = -1
        self.throughput = {"universe_steps_per_s": 0.0, "evals_per_hour_est": 0.0}
        self.stop_reason = ""
        self.last_viability: Dict[str, int] = {}

    # ==================================================================
    # utilidades de estado
    # ==================================================================
    def status_of(self, cid: str) -> Dict[str, str]:
        return {k: self.results[r].status for k, r in self.tests.get(cid, {}).items() if r in self.results}

    def result_of(self, cid: str, key: str) -> Optional[TestResult]:
        rid = self.tests.get(cid, {}).get(key)
        return self.results.get(rid) if rid else None

    def scorecard(self, cid: str) -> Dict[str, Any]:
        d = self.desc.get(cid)
        sc: Dict[str, Any] = {"candidate_id": cid}
        if d is not None:
            sc["activity"] = round(d["rel_activity"], 5)
            sc["persistence"] = round(d["persistence"], 4)
            sc["mobility"] = round(d["mobility"], 4)
            sc["complexity"] = round(0.5 * (d["temporal_complexity"] + d["spatial_complexity"]), 4)
            sc["novelty"] = round(float(self.novelty_score.get(cid, 0.0)), 4)
        for f in SCORECARD_FIELDS:
            if f in sc:
                continue
            key = "replication_channel" if f == "replication" and "replication_channel" in self.tests.get(cid, {}) else (
                "replication_memory" if f == "replication" else f)
            r = self.result_of(cid, key)
            sc[f] = STATUS_NOT_TESTED if r is None else (r.status if not np.isfinite(r.score) else f"{r.status}:{r.score:.3f}")
        sc["level"] = self.levels.get(cid, -1)
        sc["evidence"] = round(self.evidence.get(cid, 0.0), 3)
        sc["quality(fitness)"] = round(self.quality.get(cid, 0.0), 4)
        return sc

    def say(self, text: str):
        """Mensagens importantes: no painel ao vivo viram anúncios (sem poluir a saída); fora dele, print."""
        m = LIVE.monitor
        if m is not None and getattr(m, "mode", "text") == "html":
            m.announce(text)
        else:
            print(text)

    def runtime_total(self) -> float:
        return self.runtime_at_start + (time.time() - self.session_start)

    def probe_bonus(self, cid: str) -> float:
        st = self.status_of(cid)
        vals = [self.results[self.tests[cid][k]].score for k in ["memory", "channel", "causality"]
                if st.get(k) == STATUS_PASS and np.isfinite(self.results[self.tests[cid][k]].score)]
        return float(np.mean(vals)) if vals else 0.0

    # ==================================================================
    # CREATE
    # ==================================================================
    def initialize(self):
        n = self.cfg.population_size
        gs = [Genome(self.space.random_params(self.rng), generation=0, origin="random") for _ in range(n)]
        self.logger.event("genesis_created", echo=True, population=n, world_size=self.cfg.world_size,
                          channels_max=self.cfg.channels_max, profile=self.cfg.profile)
        self.evaluate_and_integrate(gs, parents={})
        self.population = [g.genome_id for g in gs]
        self.evaluate_baselines()

    def evaluate_baselines(self):
        """Baselines: lei nula, lei simples (reação-difusão), difusão pura, ruído, ancestral aleatório."""
        kinds = ["identity", "noise_replace", "diffusion", "reaction_diffusion"]
        gs = [control_genome(self.space, k, self.ve.channel_distance(self.cfg.world_size), self.cfg.world_size) for k in kinds]
        gs.append(Genome(self.space.random_params(np.random.default_rng(derive_seed(self.cfg.master_seed, "baseline-random"))),
                         origin="baseline:random"))
        ds = self.screener.screen(gs, self.ve.envs_for("evolution"), derive_seed(self.cfg.master_seed, "baselines"))
        for k, g, d in zip(kinds + ["random"], gs, ds):
            self.baselines[k] = {"descriptor": {kk: v for kk, v in d.items() if not isinstance(v, list)},
                                 "genome": genome_to_record(self.space, g), "version": VALIDATION_ENGINE_VERSION}

    # ==================================================================
    # EVOLVE
    # ==================================================================
    def select_parents(self, n: int) -> List[str]:
        rng = self.rng
        pop = [p for p in self.population if p in self.store]
        out: List[str] = []
        n_map = int(n * 0.35); n_nov = int(n * 0.2 * self.stagnation["novelty_weight"]); n_nov = min(n_nov, n // 2)
        out += [g for g in self.map.sample(rng, n_map) if g in self.store]
        nov_ids = [g for g in self.novelty.ids[-500:] if g in self.store]
        if nov_ids:
            out += [nov_ids[i] for i in rng.integers(0, len(nov_ids), size=n_nov)]
        if pop:
            obj = np.stack([self.objectives.get(p, np.zeros(7)) for p in pop])
            ranks = pareto_ranks(obj)
            nov = np.asarray([self.novelty_score.get(p, 0.0) for p in pop])
            while len(out) < n:
                a, b = rng.integers(0, len(pop), size=2)
                if ranks[a] != ranks[b]:
                    w = a if ranks[a] < ranks[b] else b
                else:
                    w = a if nov[a] >= nov[b] else b
                out.append(pop[w])
        while len(out) < n:
            out.append(str(rng.choice(list(self.store.keys()))))
        return out[:n]

    def make_offspring(self, n: int) -> Tuple[List[Genome], Dict[str, Tuple[str, str]]]:
        gen = self.counters["generation"]
        n_imm = max(0, int(round(n * self.cfg.immigrant_rate * (3 if self.stagnation["stage"] >= 3 else 1))))
        parents = self.select_parents(n - n_imm)
        kids: List[Genome] = []
        meta: Dict[str, Tuple[str, str]] = {}
        for pid in parents:
            p = self.store[pid]
            op = self.bandit.select(self.rng)
            other = None
            if op == "crossover":
                mate = self.store.get(str(self.rng.choice(self.population))) if self.population else None
                other = mate.params if mate is not None else None
                if other is None:
                    op = "gauss_small"
            params, rec = self.space.mutate(p.params, self.rng, op, self.stagnation["sigma_mult"], other)
            g = Genome(params=params, parent_ids=[pid] + ([mate.genome_id] if other is not None else []),
                       generation=gen, mutation_record=rec, origin="mutation", operator=op)
            kids.append(g); meta[g.genome_id] = (pid, op)
        for _ in range(n_imm):
            g = Genome(self.space.random_params(self.rng), generation=gen, origin="immigrant", operator="random_immigrant")
            kids.append(g)
        return kids, meta

    # ==================================================================
    # OBSERVE / MEASURE / PRESERVE
    # ==================================================================
    def evaluate_and_integrate(self, genomes: List[Genome], parents: Dict[str, Tuple[str, str]],
                               label: Optional[str] = None) -> Dict[str, Any]:
        gen = self.counters["generation"]
        t0 = time.time()
        steps0 = self.sim.steps_done
        descs = self.screener.screen(genomes, self.ve.envs_for("evolution"), derive_seed(self.cfg.master_seed, "screen", gen),
                                     label=label or f"Geração {gen}: triagem de {len(genomes)} universos")
        dt = time.time() - t0
        self.counters["evaluations"] += len(genomes)
        ds = self.sim.steps_done - steps0
        self.counters["sim_steps"] += ds
        if dt > 0:
            self.throughput["universe_steps_per_s"] = ds / dt
            self.throughput["evals_per_hour_est"] = len(genomes) / dt * 3600
        self.last_viability = dict(collections.Counter(d["viability"] for d in descs))
        V = np.stack([descriptor_vector(d) for d in descs])
        pop_vecs = np.stack([self.vec[p] for p in self.population if p in self.vec]) if self.population else None
        others = np.vstack([V, pop_vecs]) if pop_vecs is not None and len(pop_vecs) else V
        nov = self.novelty.novelty(V, others)
        surp = self.surprise.score(V)
        stats = {"new_cells": 0, "map_inserts": 0, "anomalies": 0, "viable": 0}
        viable_ids, viable_vecs, viable_nov = [], [], []
        live_progress(phase=f"Geração {gen}: arquivos, novidade, espécies", done=0, total=len(genomes),
                      detail="MAP-Elites, novelty archive, zoo, linhagem, surpresa")
        for i_, (g, d, v, nv, sp) in enumerate(zip(genomes, descs, V, nov, surp)):
            if i_ % 64 == 0:
                live_progress(done=i_)
            gid = g.genome_id
            self.store[gid] = g
            self.lineage.add(g)
            self.desc[gid] = d
            self.vec[gid] = v
            self.novelty_score[gid] = float(nv)
            obj = quality_objectives(d, self.cfg.thresholds, self.probe_bonus(gid))
            viable = d["viability"] == "VIABLE"
            q = float((obj * QUALITY_WEIGHTS).sum()) if viable else -1.0
            self.objectives[gid] = obj if viable else obj * 0 - 1
            self.quality[gid] = q
            reward = 0.0
            if viable:
                stats["viable"] += 1
                ins, new = self.map.try_insert(gid, d, q, gen)
                stats["map_inserts"] += int(ins); stats["new_cells"] += int(new)
                reward += 1.0 if new else (0.5 if ins else 0.0)
                viable_ids.append(gid); viable_vecs.append(v); viable_nov.append(nv)
                anomalous = bool(sp > self.cfg.thresholds.anomaly_surprise)
                if anomalous:
                    stats["anomalies"] += 1
                    self.anomalies.add(gid, "surprise", gen, surprise=float(sp))
                    self.logger.event("anomaly", genome_id=gid, surprise=float(sp))
                cats = Zoo.classify(d, {}, self.cfg.thresholds, anomalous)
                self.zoo.add(gid, cats, nv, gen)
                if d["causal_spread"] > 0.5 and d["lyapunov"] > self.cfg.thresholds.chaos_lyap * 2:
                    self.failures.add(gid, "high-causality chaos (triagem)", gen, lyapunov=d["lyapunov"])
                g.species_id = self.species.assign(self.space.vector(g.params), v, gen)
                level, ev = self.ladder.evaluate(d, {}, False)
                self.levels[gid] = level
                self.lineage.update(gid, quality=q, level=level, species_id=g.species_id)
            else:
                g.species_id = -1
                self.levels[gid] = -1
                self.lineage.update(gid, quality=q, level=-1)
            if gid in parents:
                pid, op = parents[gid]
                if self.quality.get(pid, -1) < q:
                    reward += 0.5
                self.bandit.update(op, min(1.0, reward))
        added = self.novelty.maybe_add(viable_ids, np.asarray(viable_vecs) if viable_vecs else np.zeros((0, V.shape[1])),
                                       np.asarray(viable_nov), self.rng)
        if viable_vecs:
            self.surprise.update(np.asarray(viable_vecs))
        stats.update({"novelty_added": len(added), "screen_seconds": dt, "n": len(genomes)})
        return stats

    def select_survivors(self, candidates: List[str]):
        cfg = self.cfg
        N = cfg.population_size
        pool = list(dict.fromkeys([c for c in candidates if c in self.store]))
        viable = [c for c in pool if self.quality.get(c, -1) >= 0]
        if len(viable) < max(4, N // 4):
            viable = sorted(pool, key=lambda c: -self.quality.get(c, -1))[:max(4, N // 4)]
        obj = np.stack([np.concatenate([self.objectives[c], [self.novelty_score.get(c, 0.0) * self.stagnation["novelty_weight"]]])
                        for c in viable])
        ranks = pareto_ranks(obj)
        nov = np.asarray([self.novelty_score.get(c, 0.0) for c in viable])
        order = np.lexsort((-nov, ranks))
        cap = max(2, int(N * cfg.species_cap_fraction))
        per_species: Dict[int, int] = {}
        chosen = []
        for i in order:
            c = viable[i]
            sid = self.store[c].species_id
            if per_species.get(sid, 0) >= cap:
                continue
            per_species[sid] = per_species.get(sid, 0) + 1
            chosen.append(c)
            if len(chosen) >= N:
                break
        if len(chosen) < N:
            rest = [viable[i] for i in order if viable[i] not in set(chosen)]
            chosen += rest[: N - len(chosen)]
        self.population = chosen

    # ==================================================================
    # PERTURB / MEASURE (experimentos)
    # ==================================================================
    def schedule_experiments(self, new_ids: List[str]):
        gen = self.counters["generation"]
        top = sorted([c for c in new_ids if self.quality.get(c, -1) >= 0], key=lambda c: -self.quality[c])[:3]
        exps = []
        cand = set(new_ids) | set(self.population[: min(len(self.population), 64)])
        for cid in cand:
            d = self.desc.get(cid)
            if d is None:
                continue
            anomalous = any(e["genome_id"] == cid for e in self.anomalies.entries[-200:])
            exps += self.scientist.propose(cid, d, self.status_of(cid), self.levels.get(cid, -1), anomalous, gen,
                                           self.vec.get(cid), is_top=cid in top)
        # replicação independente obrigatória para as descobertas mais relevantes (nível ≥ 3)
        ranked = sorted([c for c, l in self.levels.items() if l >= 3 and c in self.store],
                        key=lambda c: (-self.levels[c], -self.evidence.get(c, 0.0)))[:3]
        for cid in ranked:
            st = self.status_of(cid)
            target = max([p for p in ("memory", "channel", "transformation") if st.get(p) == STATUS_PASS],
                         key=lambda p: PROBE_LEVEL[p], default=None)
            if target is None or f"replication_{target}" in st:
                continue
            kw = {"target": target, "n_seeds": self.cfg.probe.replication_seeds}
            k_ = self.scientist.key(cid, "replication", kw)
            if k_ in self.scientist.done or k_ in self.scientist.queued:
                continue
            exps.append(Experiment(cid, "replication", kw, 100.0 + self.levels[cid],
                                   f"descoberta L{self.levels[cid]} -> replicação independente obrigatória", gen))
        self.scientist.push(exps, gen)

    def run_experiments(self, budget_s: float, max_n: int) -> int:
        t0 = time.time()
        n = 0
        while n < max_n and (time.time() - t0 < budget_s or n == 0):
            e = self.scientist.pop(self.counters["generation"])
            if e is None:
                break
            if e.candidate_id not in self.store:
                continue
            live_progress(phase=f"Geração {self.counters['generation']}: experimento {n + 1}/{max_n}", done=n, total=max_n,
                          detail=f"{e.probe} em {e.candidate_id} — {e.reason}")
            self.run_one_experiment(e)
            n += 1
        return n

    def status_key(self, e: Experiment) -> str:
        if e.probe == "replication":
            return f"replication_{e.kwargs.get('target', 'memory')}"
        return e.probe

    def run_one_experiment(self, e: Experiment):
        g = self.store[e.candidate_id]
        if e.probe == "champion_lab":
            self.champion_lab(e.candidate_id)
            self.scientist.record(e, "champion_lab", STATUS_PASS, 0.0, None)
            return
        probe = PROBE_ALIAS.get(e.probe, e.probe)
        cache_key = f"{g.hash()}|{probe}|{self.ve.PROBE_VERSIONS.get(probe)}|{json.dumps(to_jsonable(e.kwargs), sort_keys=True)}"
        seed = derive_seed(self.cfg.master_seed, "exp", e.candidate_id, e.probe, json.dumps(to_jsonable(e.kwargs), sort_keys=True))
        if cache_key in self.scientist.cache and self.scientist.cache[cache_key] in self.results:
            res = self.results[self.scientist.cache[cache_key]]
        else:
            kw = dict(e.kwargs)
            if probe not in ("replication", "ablation", "generalization", "adaptation"):
                kw.setdefault("split", "evolution")
            res = self.ve.run_probe(probe, g, seed, **kw)
            self.counters["experiments"] += 1
            self.scientist.cache[cache_key] = res.result_id
        res.details["experiment_reason"] = e.reason
        self.record_result(e.candidate_id, self.status_key(e), res)
        self.scientist.record(e, res.result_id, res.status, res.duration_s, self.vec.get(e.candidate_id))
        # experimentação ativa: o resultado gera imediatamente o próximo experimento mais informativo
        cid = e.candidate_id
        if cid in self.desc:
            anomalous = any(x["genome_id"] == cid for x in self.anomalies.entries[-200:])
            self.scientist.push(self.scientist.propose(cid, self.desc[cid], self.status_of(cid), self.levels.get(cid, -1),
                                                       anomalous, self.counters["generation"], self.vec.get(cid)),
                                self.counters["generation"])
        self.logger.event("experiment", candidate=e.candidate_id, probe=e.probe, status=res.status,
                          score=res.score, seconds=round(res.duration_s, 2), reason=e.reason)

    def record_result(self, cid: str, key: str, res: TestResult):
        if res.probe == "causality" and res.status != STATUS_PASS:
            res.details.pop("causal_map", None)      # política de armazenamento: mapas só para causalidade confirmada
        self.results[res.result_id] = res
        self.tests.setdefault(cid, {})[key] = res.result_id
        st = self.status_of(cid)
        gen = self.counters["generation"]
        # arquivo de falhas interessantes (evita redescobrir becos sem saída)
        if key == "causality" and res.details.get("chaos_suspected"):
            self.failures.add(cid, "high-causality chaos", gen, spread=res.details.get("spread_mean"))
        if key == "replication_memory" and res.status != STATUS_PASS and st.get("memory") == STATUS_PASS:
            self.failures.add(cid, "memory-like but non-reproducible", gen, pass_rate=res.score)
        if key == "channel" and res.status == STATUS_FAIL and st.get("recovery") == STATUS_PASS:
            self.failures.add(cid, "self-recovery without function", gen)
        if key == "generalization" and res.details.get("label") == "OVERFIT":
            self.failures.add(cid, "OVERFIT (falhou em entradas não vistas)", gen)
        if key == "memory" and res.details.get("leakage_detected"):
            self.failures.add(cid, "memory probe: vazamento no controle nulo (inconclusivo)", gen)
        # escada
        old = self.levels.get(cid, -1)
        level, ev = self.ladder.evaluate(self.desc.get(cid), st, st.get("replication_channel") == STATUS_PASS)
        self.levels[cid] = level
        self.evidence[cid] = evidence_score([self.results[r] for r in self.tests[cid].values() if r in self.results])
        self.lineage.update(cid, level=level)
        anomalous = any(e["genome_id"] == cid for e in self.anomalies.entries[-500:])
        if cid in self.desc:
            self.zoo.add(cid, Zoo.classify(self.desc[cid], st, self.cfg.thresholds, anomalous), self.novelty_score.get(cid, 0.0), gen)
            if self.quality.get(cid, -1) >= 0:
                obj = quality_objectives(self.desc[cid], self.cfg.thresholds, self.probe_bonus(cid))
                self.objectives[cid] = obj
                self.quality[cid] = float((obj * QUALITY_WEIGHTS).sum())
                self.map.try_insert(cid, self.desc[cid], self.quality[cid], gen)
        if level > old:
            self.audit.append({"candidate_id": cid, "from": old, "to": level, "evidence": ev, "result_id": res.result_id,
                               "generation": gen, "ts": iso_now(), "config_thresholds_hash": self.cfg.thresholds_hash()})
            if level >= 3 or (level >= 2 and level > self.best_level_global):
                self.register_discovery(cid, old, level, res, ev)
        if level >= self.cfg.thresholds.champion_level - 2:
            self.champions.add(cid, f"level>={level}", gen, level=level)

    # ==================================================================
    # DISCOVERY
    # ==================================================================
    def register_discovery(self, cid: str, old: int, new: int, res: TestResult, ev: List[str]):
        did = new_id("D")
        g = self.store[cid]
        rec = {"discovery_id": did, "candidate_id": cid, "previous_level": old, "new_level": new,
               "level_name": LEVEL_NAMES.get(new), "probe": res.probe, "result_id": res.result_id,
               "status": res.status, "score": res.score, "p_value": res.p_value, "protocol": res.protocol,
               "genome": genome_to_record(self.space, g), "evidence": ev, "generation": self.counters["generation"],
               "ts": iso_now(), "config": self.cfg.to_dict(), "engine": VALIDATION_ENGINE_VERSION,
               "global_record": new > self.best_level_global}
        self.discoveries[did] = rec
        self.counters["discovery_events"] += 1
        self.logger.event("discovery", echo=False, discovery_id=did, candidate=cid, level=new, probe=res.probe)
        if new > self.best_level_global:
            self.best_level_global = new
            self.say(discovery_report_text(rec))
            try:
                rec["bundle"] = write_evidence_bundle(self, rec)
            except Exception as e:
                self.logger.event("bundle_failed", echo=True, error=repr(e), tb=traceback.format_exc()[-1500:])

    # ==================================================================
    # CHAMPION LAB
    # ==================================================================
    def champion_lab(self, cid: str):
        if self.champion_lab_runs >= self.cfg.champion_lab_max_per_session:
            self.logger.event("champion_lab_deferred", echo=True, candidate=cid)
            return
        self.champion_lab_runs += 1
        th, ps = self.cfg.thresholds, self.cfg.probe
        g = self.store[cid]
        self.say(f"[ChampionLab] bateria final em HOLDOUT para {cid} (genoma congelado, zero evolução)")
        seed = derive_seed(self.cfg.master_seed, "champion", cid)
        battery: Dict[str, TestResult] = {}
        live_progress(phase=f"ChampionLab: {cid}", done=0, total=10, detail="bateria final em holdout")
        for i_, probe in enumerate(["causality", "memory", "channel", "transformation", "composition"]):
            live_progress(done=i_, detail=f"holdout: {probe}")
            battery[probe] = self.ve.run_probe(probe, g, derive_seed(seed, probe), split="holdout")
        live_progress(done=5, detail="holdout: generalização, robustez, replicação, ablação, análise avançada")
        battery["generalization"] = self.ve.run_probe("generalization", g, derive_seed(seed, "gen"), split="holdout", task="xor")
        battery["robustness"] = self.ve.run_probe("robustness", g, derive_seed(seed, "rob"), split="holdout")
        battery["replication"] = self.ve.run_probe("replication", g, derive_seed(seed, "rep"), target="channel",
                                                   n_seeds=ps.critical_seeds, split="holdout")
        battery["ablation"] = self.ve.run_probe("ablation", g, derive_seed(seed, "abl"), target="channel", split="holdout")
        battery["advanced"] = self.ve.run_probe("advanced", g, derive_seed(seed, "adv"), split="holdout")
        for k, r in battery.items():
            self.record_result(cid, f"holdout_{k}", r)
        gate = {
            "level": self.levels.get(cid, -1) >= th.champion_level,
            "reproducibility": battery["replication"].score >= th.champion_reproducibility,
            "robustness": battery["robustness"].score >= th.champion_robustness,
            "generalization": battery["generalization"].status == STATUS_PASS,
            "causality": battery["causality"].status == STATUS_PASS,
            "memory": battery["memory"].status == STATUS_PASS,
        }
        # dominância contra o melhor concorrente
        rivals = sorted([c for c in self.levels if c != cid and c in self.store],
                        key=lambda c: (-self.levels[c], -self.evidence.get(c, 0)))
        rival = rivals[0] if rivals else None
        dom = {"rival": rival}
        if rival is not None:
            rr = self.ve.run_probe("replication", self.store[rival], derive_seed(seed, "rival"), target="channel",
                                   n_seeds=ps.critical_seeds, split="holdout")
            a, b = np.asarray(battery["replication"].details["scores"]), np.asarray(rr.details["scores"])
            dom.update({"p": mann_whitney_greater(a, b), "effect_size": cohens_d(a, b), "rival_pass_rate": rr.score})
            dominant = (dom["p"] < th.dominance_p) and (dom["effect_size"] > th.dominance_effect)
        else:
            dominant = True
            dom.update({"p": 0.0, "effect_size": float("inf"), "note": "sem concorrentes avaliados"})
        passed = all(gate.values()) and dominant
        entry = {"candidate_id": cid, "gate": gate, "dominance": to_jsonable(dom), "passed": passed,
                 "battery": {k: r.result_id for k, r in battery.items()}, "ts": iso_now(),
                 "generation": self.counters["generation"]}
        self.champions.add(cid, "champion_lab", self.counters["generation"], **entry)
        self.logger.event("champion_lab", echo=True, candidate=cid, passed=passed, gate=gate)
        if passed:
            if self.champion is not None and self.champion["candidate_id"] != cid:
                self.champion["revoked_by"] = cid
                self.pantheon.append(self.champion)
            self.champion = entry
            self.pantheon.append(entry)
            write_champion_outputs(self, cid, battery, entry)
        else:
            reason = "NO UNIQUE CHAMPION" if all(gate.values()) and not dominant else "GATE NOT PASSED"
            self.say(f"[ChampionLab] {cid}: {reason} — {gate} | dominância: {to_jsonable(dom)}")

    # ==================================================================
    # GERAÇÃO
    # ==================================================================
    def generation_step(self) -> Dict[str, Any]:
        t0 = time.time()
        self.counters["generation"] += 1
        gen = self.counters["generation"]
        live_progress(phase=f"Geração {gen}: mutação/crossover", done=0, total=1,
                      detail=f"{self.cfg.population_size} filhos (operadores escolhidos pelo bandit)")
        kids, meta = self.make_offspring(self.cfg.population_size)
        stats = self.evaluate_and_integrate(kids, meta)
        live_progress(phase=f"Geração {gen}: seleção e Scientist", done=0, total=1, detail="Pareto + novidade + espécies")
        self.select_survivors(self.population + [k.genome_id for k in kids])
        self.species.end_generation(gen, set(self.store[c].species_id for c in self.population))
        self.schedule_experiments([k.genome_id for k in kids])
        frac = self.cfg.experiment_time_fraction
        budget = frac / max(1e-6, 1 - frac) * stats["screen_seconds"]
        stats["experiments"] = self.run_experiments(budget, self.cfg.max_experiments_per_generation)
        self.check_stagnation_and_curriculum()
        self.check_metric_gaming()
        if gen % 5 == 0:
            self.garbage_collect()
        row = self.metrics_row(stats, time.time() - t0)
        self.history.append(row)
        self.history = self.history[-20000:]
        self.logger.metrics(row)
        return row

    def best_candidate(self) -> Optional[str]:
        if not self.levels:
            return None
        return max(self.levels.keys(), key=lambda c: (self.levels[c], self.evidence.get(c, 0.0), self.quality.get(c, -1)))

    def metrics_row(self, stats: Dict[str, Any], gen_seconds: float) -> Dict[str, Any]:
        best = self.best_candidate()
        bq = max([self.quality.get(c, -1) for c in self.population], default=-1)
        pop_d = [self.desc[c] for c in self.population if c in self.desc]
        mean = lambda k: float(np.mean([d[k] for d in pop_d])) if pop_d else 0.0
        return {"generation": self.counters["generation"], "ts": iso_now(), "evaluations": self.counters["evaluations"],
                "sim_steps": self.counters["sim_steps"], "total_runtime": round(self.runtime_total(), 1),
                "gen_seconds": round(gen_seconds, 2), "best_quality": round(bq, 4),
                "map_coverage": round(self.map.coverage(), 4), "species": len(self.species.reps),
                "novelty_archive": len(self.novelty.ids), "anomalies": len(self.anomalies.entries),
                "best_level": self.levels.get(best, -1) if best else -1, "best_candidate": best,
                "viable_fraction": round(stats.get("viable", 0) / max(1, stats.get("n", 1)), 3),
                "new_cells": stats.get("new_cells", 0), "experiments": stats.get("experiments", 0),
                "mean_persistence": round(mean("persistence"), 4), "mean_causal": round(mean("causal_spread"), 4),
                "mean_tc": round(mean("temporal_complexity"), 4), "stagnation_stage": self.stagnation["stage"],
                "batch_size": self.cfg.batch_size, "universe_steps_per_s": round(self.throughput["universe_steps_per_s"], 1)}

    def check_stagnation_and_curriculum(self):
        s = self.stagnation
        bq = max([self.quality.get(c, -1) for c in self.population], default=-1)
        cov = self.map.coverage()
        bl = max(self.levels.values(), default=-1)
        improved = bq > s["best_quality"] + 1e-3 or cov > s["best_coverage"] + 1e-9 or bl > s["best_level"]
        if improved:
            s.update({"best_quality": max(bq, s["best_quality"]), "best_coverage": max(cov, s["best_coverage"]),
                      "best_level": max(bl, s["best_level"]), "since": 0})
            if s["stage"] > 0:
                s["stage"] -= 1
            s["sigma_mult"] = max(1.0, s["sigma_mult"] * 0.9)
            s["novelty_weight"] = max(1.0, s["novelty_weight"] * 0.95)
        else:
            s["since"] += 1
        if s["since"] >= self.cfg.stagnation_window:
            s["since"] = 0
            s["stage"] = min(5, s["stage"] + 1)
            st = s["stage"]
            action = {1: "aumentar mutação", 2: "aumentar pressão de novidade", 3: "criar novas espécies (imigrantes)",
                      4: "reinicializar parte da população", 5: "modificar condições ambientais"}[st]
            if st >= 1:
                s["sigma_mult"] = min(4.0, s["sigma_mult"] * 1.5)
            if st >= 2:
                s["novelty_weight"] = min(3.0, s["novelty_weight"] * 1.3)
            if st >= 4:
                k = len(self.population) // 3
                fresh = [Genome(self.space.random_params(self.rng), generation=self.counters["generation"], origin="restart")
                         for _ in range(k)]
                self.evaluate_and_integrate(fresh, {})
                keep = self.population[: len(self.population) - k]
                self.population = keep + [g.genome_id for g in fresh]
            if st >= 5:
                self.curriculum["perturbation_mult"] = min(4.0, self.curriculum["perturbation_mult"] * 1.2)
                for e in self.envs:
                    if e.split == "evolution":
                        e.ext_noise = float(min(0.02, e.ext_noise * 1.2 + 0.0005))
            self.logger.event("stagnation", echo=True, stage=st, action=action)
        # currículo: capacidades encontradas -> ambientes/probes mais difíceis
        levels = list(self.levels.values())
        if any(l >= 3 for l in levels) and self.curriculum["memory_delay_mult"] < 2.0:
            n3 = sum(1 for l in levels if l >= 3)
            if n3 >= 5:
                self.curriculum["memory_delay_mult"] = min(2.0, self.curriculum["memory_delay_mult"] * 1.25)
                self.logger.event("curriculum", echo=True, change="memory delay ↑", value=self.curriculum["memory_delay_mult"])
        if sum(1 for l in levels if l >= 4) >= 5 and self.curriculum["channel_distance_add"] < self.cfg.world_size // 4:
            self.curriculum["channel_distance_add"] += 2
            self.logger.event("curriculum", echo=True, change="channel distance ↑", value=self.curriculum["channel_distance_add"])

    def check_metric_gaming(self):
        W = self.cfg.thresholds.metric_exploit_window
        if len(self.history) < W + 1:
            return
        a, b = self.history[-W - 1], self.history[-1]
        gain = (b["best_quality"] - a["best_quality"]) / (abs(a["best_quality"]) + 1e-6)
        if gain > self.cfg.thresholds.metric_exploit_gain and b["best_level"] <= a["best_level"] and b["experiments"] >= 0:
            passes_a = sum(1 for h in self.history[-2 * W - 1:-W - 1] for _ in [0] if h.get("best_level", -1) >= 2)
            passes_b = sum(1 for h in self.history[-W:] if h.get("best_level", -1) >= 2)
            if passes_b <= passes_a:
                flag = {"generation": b["generation"], "fitness_gain": gain, "level": b["best_level"]}
                if not self.metric_flags or self.metric_flags[-1]["generation"] < b["generation"] - W:
                    self.metric_flags.append(flag)
                    self.logger.event("METRIC_EXPLOIT_SUSPECTED", echo=True, **flag)

    def garbage_collect(self):
        # elites históricas: metadados são permanentes; genomas completos para as 5000 mais recentes
        keep = set(self.population) | set(self.map.ids()) | set(h["genome_id"] for h in self.map.history[-5000:])
        keep |= set(self.novelty.ids) | set(self.zoo.ids()) | set(self.anomalies.ids()) | set(self.failures.ids())
        keep |= set(self.champions.ids()) | set(d["candidate_id"] for d in self.discoveries.values())
        keep |= set(self.tests.keys())
        keep |= set(e.candidate_id for e in self.scientist.queue)
        for gid in list(self.store.keys()):
            if gid not in keep:
                for dct in (self.store, self.desc, self.vec, self.quality, self.objectives, self.novelty_score, self.levels):
                    dct.pop(gid, None)
        self.lineage.prune(keep, self.counters["generation"])

    # ==================================================================
    # CHECKPOINT
    # ==================================================================
    def state_dict(self) -> dict:
        def rng_state():
            s = {"python": random.getstate(), "numpy_global": np.random.get_state(),
                 "lab_rng": self.rng.bit_generator.state, "torch_cpu": torch.get_rng_state()}
            if torch.cuda.is_available():
                try:
                    s["torch_cuda"] = torch.cuda.get_rng_state_all()
                except Exception as e:
                    s["torch_cuda_error"] = repr(e)
            return s
        self.counters["total_runtime"] = self.runtime_total()
        return {
            "version": GENESIS_VERSION, "checkpoint_version": CHECKPOINT_VERSION, "genesis_version": GENESIS_VERSION,
            "engine_version": VALIDATION_ENGINE_VERSION, "saved_at": iso_now(),
            "config": self.cfg.to_dict(), "hardware": self.hw, "counters": dict(self.counters),
            "population": list(self.population),
            "store": {k: genome_to_record(self.space, g) for k, g in self.store.items()},
            "desc": {k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()} for k, v in self.desc.items()},
            "vec": self.vec, "quality": self.quality, "objectives": self.objectives, "novelty_score": self.novelty_score,
            "results": {k: r.to_dict() for k, r in self.results.items()}, "tests": self.tests, "levels": self.levels,
            "evidence": self.evidence, "audit": self.audit,
            "map_elites": self.map.state_dict(), "novelty": self.novelty.state_dict(), "zoo": self.zoo.state_dict(),
            "anomalies": self.anomalies.state_dict(), "failures": self.failures.state_dict(),
            "champions": self.champions.state_dict(), "pantheon": self.pantheon, "champion": self.champion,
            "lineage": self.lineage.state_dict(), "species": self.species.state_dict(),
            "surprise": self.surprise.state_dict(), "bandit": self.bandit.state_dict(),
            "scientist": self.scientist.state_dict(), "discoveries": self.discoveries, "baselines": self.baselines,
            "history": self.history[-5000:], "stagnation": self.stagnation, "curriculum": dict(self.curriculum),
            "envs": [e.to_dict() for e in self.envs], "metric_flags": self.metric_flags,
            "best_level_global": self.best_level_global, "rng": rng_state(), "changelog": GENESIS_CHANGELOG,
        }

    def load_state_dict(self, s: dict):
        self.counters = dict(s["counters"])
        self.population = list(s["population"])
        self.store = {k: genome_from_record(self.space, v) for k, v in s["store"].items()}
        self.desc = s["desc"]; self.vec = {k: np.asarray(v, np.float32) for k, v in s["vec"].items()}
        self.quality, self.objectives, self.novelty_score = s["quality"], s["objectives"], s["novelty_score"]
        self.results = {k: TestResult.from_dict(v) for k, v in s["results"].items()}
        self.tests, self.levels, self.evidence, self.audit = s["tests"], s["levels"], s["evidence"], s["audit"]
        self.map.load_state_dict(s["map_elites"]); self.novelty.load_state_dict(s["novelty"]); self.zoo.load_state_dict(s["zoo"])
        self.anomalies.load_state_dict(s["anomalies"]); self.failures.load_state_dict(s["failures"])
        self.champions.load_state_dict(s["champions"]); self.pantheon = list(s["pantheon"]); self.champion = s["champion"]
        self.lineage.load_state_dict(s["lineage"]); self.species.load_state_dict(s["species"])
        self.surprise.load_state_dict(s["surprise"]); self.bandit.load_state_dict(s["bandit"])
        self.scientist.load_state_dict(s["scientist"]); self.discoveries = s["discoveries"]; self.baselines = s["baselines"]
        self.history = list(s["history"]); self.stagnation = s["stagnation"]
        self.curriculum.clear(); self.curriculum.update(s["curriculum"])
        self.envs[:] = [Environment(**e) for e in s["envs"]]
        self.metric_flags = s.get("metric_flags", []); self.best_level_global = s.get("best_level_global", -1)
        self.runtime_at_start = float(self.counters.get("total_runtime", 0.0))
        self.session_start = time.time()
        r = s["rng"]
        random.setstate(r["python"]); np.random.set_state(r["numpy_global"])
        self.rng = np.random.default_rng(); self.rng.bit_generator.state = r["lab_rng"]
        torch.set_rng_state(r["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in r:
            try:
                torch.cuda.set_rng_state_all(r["torch_cuda"])
            except Exception as e:
                self.logger.event("cuda_rng_restore_failed", echo=True, error=repr(e))

    def save_checkpoint(self, reason: str = "periodic") -> str:
        live_progress(phase="Salvando checkpoint", done=0, total=1, detail=f"{reason} → {self.paths.checkpoints}")
        p = self.ckpt.save(self.state_dict())
        live_progress(done=1)
        self.last_checkpoint_ts = time.time()
        self.logger.event("checkpoint", reason=reason, path=p, generation=self.counters["generation"])
        return p
