# %% [markdown]
# ## 15b — UNIVERSE ENGINE: orquestração multiescala, fidelidade adaptativa, forks contrafactuais
# Agenda temporal (ticks): sinais/ação/física de corpos **todo tick** · ambiente a cada `env_every` (chunks inativos
# com período 4× maior) · desenvolvimento/medição de corpos a cada `development_every` · fidelidade a cada
# `fidelity_every` · agregados F0 a cada `aggregate_every` · social a cada `social_every` · cultura a cada
# `culture_every` · civilização/história a cada `civ_every` (eventos discretos).
# Fidelidade científica é decisão do agendador/Scientist (novidade, ancestralidade, raridade, foco científico);
# seleção pelo usuário só promove com registro explícito — a câmera nunca altera a ciência nem o RNG.
# Escalas: F0 agregados por região ↔ F1 agentes leves ↔ F2 corpo grosso (D−1) ↔ F3 corpo completo ↔ F4 foco (gravação).

# %%
class Profiler:
    def __init__(self):
        self.ms: Dict[str, float] = collections.defaultdict(float)
        self.calls: Dict[str, int] = collections.defaultdict(int)
        self.total: Dict[str, float] = collections.defaultdict(float)

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000
            self.ms[name] = 0.95 * self.ms[name] + 0.05 * dt if self.calls[name] else dt
            self.calls[name] += 1
            self.total[name] += dt / 1000

    def report(self) -> Dict[str, Any]:
        tot = sum(self.total.values()) or 1.0
        return {k: {"ema_ms": round(self.ms[k], 3), "share": round(self.total[k] / tot, 4), "calls": self.calls[k]}
                for k in sorted(self.total, key=lambda x: -self.total[x])}


FORK_INTERVENTIONS = {"block_action", "mute_all", "freeze_plasticity", "resource_scale", "remove_structures", "none"}


class WorldFork:
    """Linha do tempo paralela (A/B) a partir do estado atual. Mesmas leis; motor de agentes próprio; sem ciência;
    marcada como intervenção. Usa números aleatórios comuns (mesma semente) para isolar o efeito da intervenção."""

    def __init__(self, uni: "UniverseEngine", label: str, intervention: Dict[str, Any]):
        self.u, self.label, self.iv = uni, label, intervention
        cfg = uni.cfg
        self.ws = uni.ws.clone()
        self.pool = AgentPool(uni.pool.cap, uni.L, cfg, uni.device)
        st = uni.pool.state_dict(); self.pool.load_state_dict(st)
        self.ledger = StructureLedger(min(cfg.max_blocks, 200000), uni.device)
        self.ids = IdCounter(10 ** 13 + derive_seed("fork", label) * 1000)
        self.engine = AgentEngine(cfg, uni.L, uni.we, uni.matter, uni.mind, uni.device)
        self.gen = torch.Generator(device=uni.device); self.gen.manual_seed(derive_seed("fork_rng", uni.tick_count))
        self.settle = SettlementEngine(cfg, uni.device)
        self.tick = uni.tick_count
        self.metrics: List[Dict[str, Any]] = []
        self.eaten = 0.0
        if intervention.get("kind") == "remove_structures":
            built = (self.ws.struct_h > 0) & (self.ws.struct_integrity < 1e8)
            self.ws.deposit = self.ws.deposit + self.ws.struct_h.clamp(min=0).float() * built
            self.ws.struct_h = torch.where(built, torch.zeros_like(self.ws.struct_h), self.ws.struct_h)
            self.ws.deposit = self.ws.deposit + self.ws.filled.float(); self.ws.filled[:] = False
            self.ws.struct_dirty = True
        if intervention.get("kind") == "resource_scale":
            self.ws.regen_mult = torch.full_like(self.ws.resource, float(np.clip(intervention.get("value", 1.0), 0, 3)))

    def ctrl(self) -> Optional[ArenaControl]:
        k = self.iv.get("kind")
        n = self.pool.cap
        if k == "block_action":
            b = torch.zeros(n, N_ACTIONS, dtype=torch.bool, device=self.u.device)
            b[:, ACTION_NAMES.index(self.iv["channel"])] = True
            return ArenaControl(action_block=b)
        if k == "mute_all":
            return ArenaControl(mute=torch.ones(n, dtype=torch.bool, device=self.u.device))
        if k == "freeze_plasticity":
            return ArenaControl(iv=MindIntervention(freeze_plasticity=torch.ones(n, dtype=torch.bool, device=self.u.device)))
        return None

    def step(self, n: int):
        cfg = self.u.cfg
        for _ in range(n):
            st = self.engine.step(self.ws, self.pool, self.tick, self.gen, None, self.ids, self.ledger, ctrl=self.ctrl(),
                                  record_interactions=False)
            self.eaten += st["eaten"]
            if self.tick % cfg.env_every == 0:
                iy, ix = WorldEngine.tile_index(self.ws, self.pool.pos[self.pool.alive])
                self.ws.tick = self.tick
                self.u.we.env_step(self.ws, self.gen, torch.stack([iy, ix]))
            self.tick += 1

    def measure(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.tick)
        sm = self.settle.update(self.ws, self.pool, self.tick, None, rng)
        m = {"tick": self.tick, "population": self.pool.n_alive(), "blocks": sm["blocks_total"], "settlements": sm["settlements"],
             "enclosures": sm["enclosures_total"], "bridges": int(self.ws.filled.sum()), "eaten": self.eaten,
             "tech": self.engine.tech_counts.cpu().numpy().tolist(),
             "mean_energy": float(self.pool.energy[self.pool.alive].mean()) if self.pool.n_alive() else 0.0}
        self.metrics.append(m)
        return m


class UniverseEngine:
    def __init__(self, cfg: CivConfig, paths: Optional[CivPaths], device: torch.device, logger: Optional[EventLogger],
                 hw: Optional[Dict[str, Any]] = None):
        self.cfg, self.paths, self.device, self.logger, self.hw = cfg, paths, device, logger, hw or {}
        L = self.L = GenomeLayout(cfg).to(device)
        self.we = WorldEngine(cfg, device)
        self.matter = MatterEngine(cfg, device)
        self.dev = DevelopmentEngine(cfg, L, device)
        cfg_c = copy.deepcopy(cfg); cfg_c.dev_lattice = max(3, cfg.dev_lattice - 1)
        self.dev_coarse = DevelopmentEngine(cfg_c, L, device)
        self.body = BodyEngine(cfg, self.dev, device)
        self.body_coarse = BodyEngine(cfg, self.dev_coarse, device)
        self.mind = SignalEngine(cfg, L, device)
        self.agents = AgentEngine(cfg, L, self.we, self.matter, self.mind, device)
        self.runner = ArenaRunner(cfg, L, self.we, self.matter, self.mind, self.dev, self.body, device)
        self.qd = QDEvolution(cfg, L, self.runner, device, logger)
        self.sentinel = EthicalSentinel(cfg)
        self.affect = AffectEngine(cfg, L, self.runner, self.sentinel, device)
        self.validation = ValidationEngine(cfg, L, self.runner, self.affect, device)
        self.archive = ArchiveEngine(paths, cfg)
        self.scientist = ScientistEngine(cfg, L, self.validation, self.archive, self.qd, logger, device)
        self.social = SocialEngine(cfg)
        self.culture = CultureEngine(cfg, L)
        self.settlements = SettlementEngine(cfg, device)
        self.civ = CivilizationEngine(cfg, L)
        self.history = HistoryEngine(paths)
        self.lineage = LineageStore(paths)
        self.ids = IdCounter(1)
        self.pool = AgentPool(cfg.max_agents, L, cfg, device)
        self.ledger = StructureLedger(cfg.max_blocks, device)
        nb = cfg.max_bodies
        sub = cfg.body_substeps
        self.bodies = {2: BodyRegistry(cfg, self.dev_coarse, self.body_coarse, max(2, int(nb * 0.6)), sub),
                       3: BodyRegistry(cfg, self.dev, self.body, max(2, int(nb * 0.3)), sub),
                       4: BodyRegistry(cfg, self.dev, self.body, max(2, int(nb * 0.1)), sub, record=True)}
        self.ws: Optional[WorldState] = None
        self.f0: Optional[RegionAggregates] = None
        self.gen = torch.Generator(device=device); self.gen.manual_seed(derive_seed("world_rng", cfg.master_seed))
        self.inst_rng = np.random.default_rng(derive_seed("instruments", cfg.master_seed) % (2 ** 32))
        self.tick_count = 0
        self.created = iso_now()
        self.total_runtime = 0.0
        self.sessions = 0
        self.window = collections.Counter()
        self.demography = RingBuffer(2000)
        self.focus: Dict[str, Dict[str, Any]] = {}
        self.user_focus_aid: Optional[int] = None
        self.interventions: List[Dict[str, Any]] = []
        self.fidelity_log = collections.Counter()
        self.prof = Profiler()
        self.paused = False
        self.last = {"social": {}, "comm": {}, "culture": {}, "settle": {}, "civ": {}}
        self.forks: Dict[str, Dict[str, Any]] = {}
        self.events_out: "collections.deque" = collections.deque(maxlen=500)
        self.legacy_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ criação
    def create_world(self, bootstrap_seconds: float = 30.0):
        cfg = self.cfg
        seed = derive_seed("world", cfg.master_seed)
        self.ws = self.we.generate(seed)
        self.f0 = RegionAggregates(cfg, self.ws.H, self.ws.W, self.L.P)
        t0 = time.time()
        self.qd.init_islands()
        while time.time() - t0 < bootstrap_seconds or self.qd.generations < 2 * len(self.qd.islands):
            drive(self.qd.generation_gen())
            live_progress("bootstrap QD", time.time() - t0, bootstrap_seconds)
            if time.time() - t0 > 4 * bootstrap_seconds:
                break
        n = cfg.initial_agents
        el = self.qd.elites_for_founding(max(1, n // 2))
        parts = []
        if len(el):
            E = torch.as_tensor(el, device=self.device)
            reps = E[torch.randint(0, E.shape[0], (int(n * 0.8),), generator=self.gen, device=self.device)]
            parts.append(self.L.mutate(reps, self.gen, 0.5))
        parts.append(self.L.random(n - sum(p.shape[0] for p in parts), self.gen))
        G = torch.cat(parts)[:n]
        slots = self.agents.spawn_founders(self.ws, self.pool, n, self.gen, self.ids, 0, self.lineage, genomes=G)
        self._measure(slots)
        self.history.add("world_created", 0, seed=seed, size=[self.ws.W, self.ws.H], founders=int(slots.numel()),
                         qd_generations=self.qd.generations, bootstrap_seconds=round(time.time() - t0, 1),
                         note="fundadores: 80% elites QD mutadas + 20% genomas aleatórios")
        if self.logger:
            self.logger.event("world_created", echo=True, seed=seed, founders=int(slots.numel()), W=self.ws.W, H=self.ws.H)

    def _measure(self, slots: torch.Tensor, chunk: int = 256):
        """Desenvolvimento + teste físico de marcha (resumo F1) para agentes ainda não medidos."""
        for i in range(0, slots.numel(), chunk):
            s = slots[i:i + chunk]
            cap = self.body.measure_capability(self.pool.genome[s], (self.pool.aid[s] % 100003))
            self.pool.body_cap[s] = cap
            self.pool.body_measured[s] = True

    # ------------------------------------------------------------------ passo do mundo
    def tick(self):
        cfg, t, ws, pool = self.cfg, self.tick_count, self.ws, self.pool
        with self.prof("agents+signals+bodies"):
            st = self.agents.step(ws, pool, t, self.gen, self.lineage, self.ids, self.ledger, bodies=self.bodies)
        for k, v in st.items():
            if k != "alive":
                self.window[k] += v
        self.window["agent_ticks"] += st["alive"]
        if t % cfg.env_every == 0:
            with self.prof("environment"):
                iy, ix = WorldEngine.tile_index(ws, pool.pos[pool.alive])
                ws.tick = t
                self.we.env_step(ws, self.gen, torch.stack([iy, ix]))
        if t % cfg.development_every == 0:
            with self.prof("development"):
                pend = torch.nonzero(pool.alive & ~pool.body_measured).flatten()
                k = 64 if self.device.type == "cuda" else 12
                if pend.numel():
                    self._measure(pend[-k:])
        if t % cfg.fidelity_every == 0:
            with self.prof("fidelity"):
                self._fidelity()
        if t % cfg.aggregate_every == 0:
            with self.prof("aggregate F0"):
                self._aggregate()
        if t % cfg.social_every == 0:
            with self.prof("social"):
                self.social.ingest(self.agents.interactions.drain(), t)
                self.culture.ingest_tech(self.agents.tech_events, self.social, t)
                self.sentinel.observe(pool, self.L, t)
            if t % (cfg.social_every * 5) == 0:
                with self.prof("social groups"):
                    self.last["social"] = self.social.update_groups(pool, t, self.inst_rng)
        if t % cfg.culture_every == 0:
            with self.prof("culture+communication"):
                self.last["culture"] = self.culture.update(pool, self.social, t, self.inst_rng)
                c = self.social.communication(pool, ws, self.inst_rng, self.cfg.probe.permutations, t)
                if c:
                    self.last["comm"] = c
        if t % cfg.civ_every == 0 and t > 0:
            with self.prof("civilization"):
                self._civilization(t)
        if t % cfg.immigrant_every == 0 and t > 0:
            self._immigrants(t)
        if pool.n_alive() < max(8, cfg.initial_agents // 20):
            self._recover(t)
        if t % 200 == 0:
            with self.prof("candidates"):
                self._harvest_candidates(t)
        self.tick_count += 1
        ws.tick = self.tick_count

    def _civilization(self, t: int):
        cfg = self.cfg
        settle = self.settlements.update(self.ws, self.pool, t, self.social, self.inst_rng)
        self.last["settle"] = settle
        new_inv = sum(1 for e in self.culture.new_events if e["type"] == "invention")
        demo = {"births": self.window["births"], "deaths": self.window["deaths"], "eaten": self.window["eaten"],
                "agent_ticks": self.window["agent_ticks"], "new_inventions": new_inv}
        validated = dict(self.scientist.validated_any)
        civ = self.civ.evaluate(t, self.pool, self.f0.total() if self.f0 else 0.0, self.last["social"], self.last["comm"],
                                self.last["culture"], settle, validated, self.inst_rng, demo)
        self.last["civ"] = civ
        self.demography.push({"tick": t, "population": self.pool.n_alive(), "f0": self.f0.total() if self.f0 else 0,
                              **{k: self.window[k] for k in ("births", "deaths", "deaths_starvation", "deaths_exposure",
                                                             "deaths_age", "deaths_conflict", "placed", "picked", "given", "taken")},
                              "eaten": round(self.window["eaten"], 3)})
        self.window = collections.Counter()
        for e in self.history.observe(t, self.pool, civ.get("species_estimate", 0)):
            self.events_out.append(e)
        for src in (self.culture.drain_events(), self.settlements.drain_events(), self.civ.drain_events(),
                    self.sentinel.drain_events()):
            for e in src:
                et = e.get("type", "event")
                pos = e.get("pos")
                ev = self.history.add({"invention": "technology_discovery"}.get(et, et), e.get("tick", t), pos=pos,
                                      **{k: v for k, v in e.items() if k not in ("type", "tick", "pos")})
                self.events_out.append(ev)
                if et == "civilization_level":
                    meta = self.scientist.record_civilization(e["level"], t, self.civ.achieved.get(e["level"], {}).get("evidence", {}))
                    self.events_out.append(self.history.add("civilization_discovery", t, level=e["level"], discovery=meta["id"]))
        for d in self.scientist.discovery_events:
            self.events_out.append(self.history.add("science_discovery", t, **{k: v for k, v in d.items() if k != "type"}))
        self.scientist.discovery_events = []
        self.history.flush()

    # ------------------------------------------------------------------ fidelidade adaptativa
    def _fidelity(self):
        pool = self.pool
        alive = torch.nonzero(pool.alive).flatten()
        if alive.numel() == 0:
            return
        sig = pool.signature[alive]
        nov = (sig - sig.mean(0)).norm(dim=-1)
        nov = nov / nov.max().clamp(min=1e-6)
        age = pool.age[alive] / pool.age[alive].max().clamp(min=1)
        score = 0.5 * nov + 0.3 * age + 0.2 * pool.energy[alive]
        if self.focus:
            F = torch.as_tensor(np.stack([v["genome"] for v in self.focus.values()]), device=self.device)
            dmin = torch.cdist(pool.genome[alive] / (self.L.hi - self.L.lo).clamp(min=1e-6),
                               F / (self.L.hi - self.L.lo).clamp(min=1e-6)).min(1).values / math.sqrt(self.L.P)
            score = score + 2.0 * (dmin < 0.05).float()
        if self.user_focus_aid is not None:
            hit = pool.aid[alive] == self.user_focus_aid
            if bool(hit.any()):
                score = score + 5.0 * hit.float()
                self.fidelity_log["user_selected_promotion"] += int(hit.sum())
        order = alive[torch.argsort(score, descending=True)]
        caps = {4: self.bodies[4].cap, 3: self.bodies[3].cap, 2: self.bodies[2].cap}
        want: Dict[int, torch.Tensor] = {}
        i = 0
        for lvl in (4, 3, 2):
            want[lvl] = order[i:i + caps[lvl]]
            i += caps[lvl]
        max_new = 32 if self.device.type == "cuda" else 6
        for lvl in (4, 3, 2):
            reg = self.bodies[lvl]
            cur = reg.agent_slot[reg.agent_slot >= 0]
            keep = torch.isin(cur, want[lvl])
            drop_slots = torch.nonzero((reg.agent_slot >= 0) & ~torch.isin(reg.agent_slot, want[lvl])).flatten()
            if drop_slots.numel():
                a = reg.agent_slot[drop_slots]
                reg.detach(drop_slots)
                pool.body_slot[a] = -1; pool.fidelity[a] = 1
                self.fidelity_log[f"demote_F{lvl}"] += int(a.numel())
            new = want[lvl][~torch.isin(want[lvl], cur[keep])]
            new = new[pool.body_slot[new] < 0][:max_new]
            if new.numel():
                bs = reg.attach(new, pool.aid[new], pool.genome[new], pool.aid[new] % 100003)
                k = bs.numel()
                pool.body_slot[new[:k]] = bs.int(); pool.fidelity[new[:k]] = lvl
                self.fidelity_log[f"promote_F{lvl}"] += int(k)

    # ------------------------------------------------------------------ F0 ↔ F1
    def _aggregate(self):
        cfg, pool, ws = self.cfg, self.pool, self.ws
        if self.f0 is None:
            return
        self.f0.update(ws, cfg.aggregate_every, self.inst_rng)
        n = pool.n_alive()
        if n > cfg.f1_budget_fraction * pool.cap:
            # agregação: regiões mais densas, nunca agentes com corpo/foco
            k = int(n - 0.8 * pool.cap)
            idx = torch.nonzero(pool.alive & (pool.fidelity <= 1)).flatten()
            if idx.numel() == 0:
                return
            reg = torch.as_tensor(self.f0.region_of(pool.pos[idx].cpu().numpy()), device=self.device)
            dens = torch.bincount(reg, minlength=self.f0.Rh * self.f0.Rw).float()
            pick = idx[torch.argsort(dens[reg] + 0.01 * torch.rand(idx.numel(), generator=self.gen, device=self.device),
                                     descending=True)[:k]]
            regs = self.f0.region_of(pool.pos[pick].cpu().numpy())
            self.f0.absorb(regs, pool.genome[pick].cpu().numpy(), pool.energy[pick].cpu().numpy(),
                           pool.signature[pick].cpu().numpy(), self.inst_rng)
            self.agents.kill(pool, pick, self.bodies, ws)
            self.fidelity_log["aggregate_F1_to_F0"] += int(pick.numel())
        elif n < 0.5 * pool.cap and self.f0.total() >= 1:
            self._disaggregate(int(min(self.f0.total(), 0.6 * pool.cap - n, 256)))

    def _disaggregate(self, k: int):
        """Desagregação consistente: amostra genomas do reservatório da região, na própria região, conta decrementada."""
        if k <= 0:
            return 0
        f0, ws = self.f0, self.ws
        counts = f0.count.flatten()
        regs = [r for r in np.argsort(-counts) if counts[r] >= 1 and r in f0.reservoir][:64]
        made = 0
        for r in regs:
            if made >= k:
                break
            take = int(min(counts[r], k - made, 32))
            ry, rx = divmod(int(r), f0.Rw)
            y0, x0 = ry * f0.rs, rx * f0.rs
            G = torch.as_tensor(np.stack([f0.reservoir[r][i % len(f0.reservoir[r])] for i in range(take)]), device=self.device)
            G = self.L.mutate(G, self.gen, 0.3)
            slots = self.agents.spawn_founders(ws, self.pool, take, self.gen, self.ids, self.tick_count, self.lineage,
                                               genomes=G, origin="disaggregated",
                                               region=(y0, x0, min(ws.H, y0 + f0.rs), min(ws.W, x0 + f0.rs)),
                                               energy=float(np.clip(f0.energy[ry, rx], 0.3, 0.9)))
            self._measure(slots)
            f0.count[ry, rx] -= slots.numel()
            made += int(slots.numel())
        self.fidelity_log["disaggregate_F0_to_F1"] += made
        return made

    def _immigrants(self, t: int):
        G, gids = self.qd.best_genomes(max(2, self.cfg.initial_agents // 50))
        if not len(G):
            return
        slots = self.agents.spawn_founders(self.ws, self.pool, len(G), self.gen, self.ids, t, self.lineage,
                                           genomes=torch.as_tensor(G, device=self.device), origin="immigrant_qd")
        self._measure(slots)
        if slots.numel():
            self.history.add("immigration", t, n=int(slots.numel()), source="QD islands")

    def _recover(self, t: int):
        made = self._disaggregate(self.cfg.initial_agents // 4) if self.f0 is not None and self.f0.total() >= 1 else 0
        if self.pool.n_alive() + made < max(8, self.cfg.initial_agents // 20):
            G = self.qd.elites_for_founding(max(8, self.cfg.initial_agents // 4))
            if len(G):
                slots = self.agents.spawn_founders(self.ws, self.pool, len(G), self.gen, self.ids, t, self.lineage,
                                                   genomes=self.L.mutate(torch.as_tensor(G, device=self.device), self.gen, 0.5),
                                                   origin="immigrant_qd")
                self._measure(slots)
                made += int(slots.numel())
        if made:
            self.history.add("recovery_injection", t, n=made, note="população abaixo do piso: desagregação F0 + imigrantes QD")

    def _harvest_candidates(self, t: int):
        pool = self.pool
        idx = torch.nonzero(pool.alive & (pool.age > 100)).flatten()
        sc = self.scientist
        if idx.numel():
            rate = pool.stats[idx, 0] / pool.age[idx].clamp(min=1)
            top = idx[torch.argsort(rate, descending=True)[:4]]
            sig = pool.signature[idx]
            nov = (sig - sig.mean(0)).norm(dim=-1)
            topn = idx[torch.argsort(nov, descending=True)[:2]]
            for s, kind in [(int(x), "intake") for x in top.tolist()] + [(int(x), "novelty") for x in topn.tolist()]:
                aid = int(pool.aid[s])
                c = sc.add_candidate(pool.genome[s].cpu().numpy(), f"world:agent:{aid}:{kind}", t,
                                     fitness=float(pool.stats[s, 0] / pool.age[s].clamp(min=1) * 100),
                                     novelty=float(nov.max()) if kind == "novelty" else 0.0,
                                     importance=math.log1p(self.lineage.children_count.get(aid, 0)))
        G, gids = self.qd.best_genomes(3)
        for g, gid in zip(G, gids):
            isl = next((i.iid for i in self.qd.islands if gid in i.gids), -1)
            f = next((float(i.fit[i.gids.index(gid)]) for i in self.qd.islands if gid in i.gids), 0.0)
            sc.add_candidate(g, f"qd:island{isl}:{gid}", t, fitness=f, island=isl)
        sc.watch_fitness(max((i.best for i in self.qd.islands), default=0.0))
        for c in list(self.archive.zoo.values()):
            if c.passed and c.cid not in self.focus and len(self.focus) < 16:
                self.focus[c.cid] = {"genome": c.genome, "reason": "validated_candidate"}

    # ------------------------------------------------------------------ forks contrafactuais
    def counterfactual_gen(self, intervention: Dict[str, Any], ticks: int, label: str = ""):
        """Linha A (sem mudança) × linha B (intervenção) a partir do mesmo estado. Gerador (rende a cada 25 ticks)."""
        kind = intervention.get("kind")
        if kind not in FORK_INTERVENTIONS:
            raise ValueError("intervenção não permitida")
        if kind == "block_action" and intervention.get("channel") not in ACTION_NAMES:
            raise ValueError("canal inválido")
        ticks = int(np.clip(ticks, 25, 5000))
        fid = f"FORK-{len(self.forks) + 1:04d}-{uuid.uuid4().hex[:6]}"
        A, B = WorldFork(self, f"{fid}-A", {"kind": "none"}), WorldFork(self, f"{fid}-B", intervention)
        rec = {"id": fid, "from_tick": self.tick_count, "intervention": intervention, "ticks": ticks, "label": label,
               "status": "running", "A": [], "B": [], "ts": iso_now(), "marked": "EXTERNAL INTERVENTION (fork only)"}
        self.forks[fid] = rec
        self.interventions.append({"tick": self.tick_count, "fork": fid, "intervention": intervention, "ts": iso_now()})
        done = 0
        while done < ticks:
            step = min(25, ticks - done)
            A.step(step); B.step(step); done += step
            rec["A"].append(A.measure()); rec["B"].append(B.measure())
            yield
        a, b = rec["A"][-1], rec["B"][-1]
        rec["diff"] = {k: (b[k] - a[k]) if isinstance(a[k], (int, float)) else None for k in a if k != "tick"}
        rec["status"] = "done"
        diverge = (A.ws.struct_h != B.ws.struct_h) | (A.ws.filled != B.ws.filled)
        rec["divergent_tiles"] = int(diverge.sum())
        rec["divergence_map"] = F.avg_pool2d(diverge.float()[None, None], 4)[0, 0].cpu().numpy().round(3)
        if self.paths is not None:
            with open(os.path.join(self.paths.civilizations, f"{fid}.json"), "w") as f:
                json.dump(to_jsonable({k: v for k, v in rec.items()}), f)
        self.history.add("counterfactual_fork", self.tick_count, fork=fid, intervention=intervention, diff=rec["diff"])
        return rec

    # ------------------------------------------------------------------ estado (para o CheckpointEngine)
    def state_dict(self) -> Dict[str, Any]:
        return {
            "meta": {"tick": self.tick_count, "created": self.created, "total_runtime": self.total_runtime,
                     "sessions": self.sessions, "ids": self.ids.value, "fidelity_log": dict(self.fidelity_log),
                     "interventions": self.interventions[-500:], "versions": ENGINE_VERSIONS,
                     "versions_hash": ENGINE_VERSIONS_HASH, "config": self.cfg.to_dict(), "legacy": self.legacy_summary,
                     "focus": {k: {"genome": v["genome"], "reason": v["reason"]} for k, v in self.focus.items()},
                     "forks": {k: {kk: vv for kk, vv in v.items() if kk != "divergence_map"} for k, v in list(self.forks.items())[-20:]}},
            "world": self.ws.state_dict(), "agents": self.pool.state_dict(), "ledger": self.ledger.state_dict(),
            "lineage": self.lineage.state_dict(), "f0": self.f0.state_dict() if self.f0 else None,
            "social": self.social.state_dict(), "culture": self.culture.state_dict(),
            "settlements": self.settlements.state_dict(), "civ": self.civ.state_dict(), "history": self.history.state_dict(),
            "qd": self.qd.state_dict(), "archive": self.archive.state_dict(), "scientist": self.scientist.state_dict(),
            "sentinel": self.sentinel.state_dict(),
            "affect": {"directions": self.affect.directions},
            "rng": {"world": self.gen.get_state().cpu().numpy(), "instruments": self.inst_rng.bit_generator.state},
            "demography": list(self.demography.items),
        }

    def load_state_dict(self, s: Dict[str, Any]):
        m = s["meta"]
        self.tick_count, self.created = int(m["tick"]), m["created"]
        self.total_runtime, self.sessions = float(m["total_runtime"]), int(m["sessions"])
        self.ids.value = int(m["ids"]); self.fidelity_log = collections.Counter(m.get("fidelity_log", {}))
        self.interventions = list(m.get("interventions", [])); self.legacy_summary = m.get("legacy")
        self.focus = {k: {"genome": np.asarray(v["genome"], np.float32), "reason": v["reason"]} for k, v in m.get("focus", {}).items()}
        self.forks = dict(m.get("forks", {}))
        self.ws = WorldState.from_state_dict(s["world"], self.device)
        ag = s["agents"]
        n = int(ag["n"])
        if n > self.pool.cap:
            # perfil menor que o do checkpoint: excedente vai para F0 (contagem e genomas preservados — bookkeeping correto)
            extra = {k: np.asarray(v)[self.pool.cap:] for k, v in ag.items() if isinstance(v, np.ndarray) and len(v) == n}
            ag = {k: (np.asarray(v)[: self.pool.cap] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in ag.items()}
            ag["n"] = self.pool.cap
            self._pending_f0 = extra
        self.pool.load_state_dict(ag)
        self.pool.body_slot[:] = -1
        self.pool.fidelity[self.pool.alive] = 1                    # corpos são reconstruídos pelo desenvolvimento (determinístico)
        self.ledger.load_state_dict(s["ledger"])
        self.lineage.load_state_dict(s["lineage"])
        self.f0 = RegionAggregates(self.cfg, self.ws.H, self.ws.W, self.L.P)
        if s.get("f0"):
            self.f0.load_state_dict(s["f0"])
        if getattr(self, "_pending_f0", None):
            ex = self._pending_f0
            self.f0.absorb(self.f0.region_of(ex["pos"]), ex["genome"], ex["energy"], ex["signature"], self.inst_rng)
            self._pending_f0 = None
        self.social.load_state_dict(s["social"]); self.culture.load_state_dict(s["culture"])
        self.settlements.load_state_dict(s["settlements"]); self.civ.load_state_dict(s["civ"])
        self.history.load_state_dict(s["history"]); self.qd.load_state_dict(s["qd"])
        self.archive.load_state_dict(s["archive"]); self.scientist.load_state_dict(s["scientist"])
        self.sentinel.load_state_dict(s["sentinel"])
        self.affect.directions = {k: np.asarray(v, np.float32) for k, v in s.get("affect", {}).get("directions", {}).items()}
        try:
            self.gen.set_state(torch.as_tensor(np.asarray(s["rng"]["world"]), dtype=torch.uint8))
            self.inst_rng.bit_generator.state = s["rng"]["instruments"]
        except Exception as e:
            if self.logger:
                self.logger.event("rng_restore_warning", error=repr(e))
        for x in s.get("demography", []):
            self.demography.push(x)
        self.ws.tick = self.tick_count

    # ------------------------------------------------------------------ resumo
    def status(self) -> Dict[str, Any]:
        civ = self.last.get("civ", {})
        return {"tick": self.tick_count, "population": self.pool.n_alive(), "f0_population": float(self.f0.total()) if self.f0 else 0.0,
                "bodies": {f"F{k}": int((r.agent_slot >= 0).sum()) for k, r in self.bodies.items()},
                "lineage_total": self.lineage.total, "civ_level": self.civ.level,
                "civ_level_name": C_LEVELS[self.civ.level] if self.civ.level >= 0 else "—",
                "h_level": self.scientist.best_level.get("H", -1), "a_level": self.scientist.best_level.get("A", -1),
                "settlements": civ.get("settlements", 0), "blocks": civ.get("blocks", 0.0),
                "qd_generations": self.qd.generations, "discoveries": len(self.archive.discoveries),
                "paused": self.paused, "world_age": self.tick_count, "created": self.created}
