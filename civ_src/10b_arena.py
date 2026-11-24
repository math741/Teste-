# %% [markdown]
# ## 10b — LAB ARENAS: ensaios padronizados com a mesma física do planeta
# Os probes e a evolução QD não usam um "simulador de teste" diferente: usam o **mesmo** `AgentEngine`, o mesmo
# `WorldEngine` e o mesmo `MatterEngine`, em sub-mundos separados por paredes. Layouts são determinísticos por
# (semente, arena) → condições pareadas (tratamento × controle) veem exatamente o mesmo mundo e as mesmas posições
# iniciais. Execução em **gerador** (`run_gen`): o laço vivo intercala fatias de ciência com o mundo principal, então o
# observatório 3D nunca congela durante experimentos.

# %%
@dataclass
class ArenaSpec:
    task: str = "forage"
    n_patches: int = 3
    patch_value: float = 1.0
    patch_radius: int = 1
    regen: bool = False
    rich: int = 0                       # manchas cooperativas (extração plena só com ≥2 consumidores)
    water_barrier: bool = False         # faixa de água entre início e recurso (exige aterro para cruzar)
    deposit: float = 0.0                # matéria por tile numa zona de depósito perto do início
    storm: float = 0.0                  # exposição constante
    cold: float = 0.0                   # 0..1 reduz a temperatura
    day_night: bool = False
    hidden_side: bool = False           # recurso num lado sorteado por arena (tarefas de pista/memória)
    resource_floor: float = 0.0         # recurso basal disperso
    spawn_side: str = "center"          # center | left


@dataclass
class Trial:
    ws: WorldState
    pool: AgentPool
    ledger: StructureLedger
    gen: torch.Generator
    spec: ArenaSpec
    n_arenas: int
    per_arena: int
    arena_of: torch.Tensor              # [n] índice da arena de cada agente
    origins: torch.Tensor               # [n_arenas, 2] (y0, x0)
    label: torch.Tensor                 # [n_arenas] rótulo oculto (ex.: lado do recurso)
    seed: int
    tick: int = 0
    patch_tiles: Optional[torch.Tensor] = None   # [n_arenas, k, 2] (y, x) centros das manchas
    barrier_x: int = -1

    @property
    def n(self) -> int:
        return self.n_arenas * self.per_arena


class ArenaRunner:
    LAB_ID_BASE = 10 ** 12              # IDs de laboratório nunca colidem com IDs do mundo

    def __init__(self, cfg: CivConfig, layout: GenomeLayout, we: WorldEngine, matter: MatterEngine, mind: MindEngine,
                 dev: DevelopmentEngine, body: BodyEngine, device: torch.device):
        self.cfg, self.L, self.we, self.matter, self.mind, self.dev, self.body, self.device = \
            cfg, layout, we, matter, mind, dev, body, device
        self.engine = AgentEngine(cfg, layout, we, matter, mind, device)
        self.ids = IdCounter(self.LAB_ID_BASE)
        self.cap_cache: "collections.OrderedDict[str, torch.Tensor]" = collections.OrderedDict()
        self.current: Optional[Trial] = None
        self.current_name = ""
        self.ticks_run = 0

    # ------------------------------------------------------------------ capacidade física (F1) com cache
    def capability(self, G: torch.Tensor, seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        keys = [hashlib.sha1(g.detach().cpu().numpy().tobytes()).hexdigest()[:20] for g in G]
        miss = [i for i, k in enumerate(keys) if k not in self.cap_cache]
        if miss:
            mi = torch.tensor(miss, device=self.device)
            sd = seeds[mi] if seeds is not None else torch.tensor([int(keys[i][:8], 16) % 100003 for i in miss], device=self.device)
            caps = self.body.measure_capability(G[mi], sd)
            for j, i in enumerate(miss):
                self.cap_cache[keys[i]] = caps[j].detach()
            while len(self.cap_cache) > 20000:
                self.cap_cache.popitem(last=False)
        return torch.stack([self.cap_cache[k] for k in keys]).to(self.device)

    # ------------------------------------------------------------------ construção do ensaio
    def build(self, genomes: torch.Tensor, spec: ArenaSpec, seed: int, per_arena: int = 1,
              n_arenas: Optional[int] = None, body_cap: Optional[torch.Tensor] = None, energy: float = 0.8,
              W_override: Optional[torch.Tensor] = None, x_override: Optional[torch.Tensor] = None,
              label: Optional[torch.Tensor] = None) -> Trial:
        """genomes: [n_arenas*per_arena, P] (agentes da arena a ficam em [a*per_arena:(a+1)*per_arena])."""
        cfg, d = self.cfg, self.device
        n = genomes.shape[0]
        n_arenas = n_arenas or max(1, n // per_arena)
        assert n == n_arenas * per_arena, "genomes deve ter n_arenas*per_arena linhas"
        S = cfg.probe.arena_size
        ws = self.we.make_arena_world(n_arenas, S, seed, base_resource=spec.resource_floor, day_night=spec.day_night,
                                      weather=False)
        ws.flags["regen"] = spec.regen
        if spec.regen:
            ws.fertility = torch.where(ws.struct_h == 0, torch.full_like(ws.fertility, 0.4), ws.fertility)
        rng = np.random.default_rng(derive_seed("arena_layout", seed) % (2 ** 32))
        origins = torch.tensor(ws.arena["origins"], device=d, dtype=torch.long)
        c = S // 2
        lab = label.clone().to(d) if label is not None else torch.as_tensor(rng.integers(0, 2, n_arenas), device=d)
        patches = []
        barrier_x = -1
        if spec.water_barrier:
            barrier_x = c + 1
        for a in range(n_arenas):
            oy, ox = ws.arena["origins"][a]
            pts = []
            if spec.hidden_side:
                side = int(lab[a])
                px = c - 5 if side == 0 else c + 5
                pts.append((int(rng.integers(c - 3, c + 4)), px))
            else:
                for _ in range(spec.n_patches):
                    for _try in range(50):
                        py, px = int(rng.integers(1, S - 1)), int(rng.integers(1, S - 1))
                        if spec.water_barrier and px <= barrier_x + 2:
                            continue
                        if (py - c) ** 2 + (px - c) ** 2 >= 9:
                            break
                    pts.append((py, px))
            patches.append(pts)
            for (py, px) in pts:
                y0, y1 = oy + max(0, py - spec.patch_radius), oy + min(S, py + spec.patch_radius + 1)
                x0, x1 = ox + max(0, px - spec.patch_radius), ox + min(S, px + spec.patch_radius + 1)
                ws.resource[y0:y1, x0:x1] = spec.patch_value
                if spec.regen:
                    ws.fertility[y0:y1, x0:x1] = 1.0
            for _ in range(spec.rich):
                py, px = int(rng.integers(2, S - 2)), int(rng.integers(2, S - 2))
                ws.rich[oy + py - 1:oy + py + 2, ox + px - 1:ox + px + 2] = True
                ws.resource[oy + py - 1:oy + py + 2, ox + px - 1:ox + px + 2] = spec.patch_value
            if spec.water_barrier:
                ws.is_water[oy:oy + S, ox + barrier_x:ox + barrier_x + 2] = True
            if spec.deposit > 0:
                sx = 2 if spec.spawn_side == "left" else c - 2
                ws.deposit[oy + c - 2:oy + c + 3, ox + max(1, sx - 1):ox + sx + 2] = spec.deposit
        if spec.storm > 0:
            ws.storm = (ws.struct_h < self.cfg.max_stack).float() * spec.storm
        if spec.cold > 0:
            ws.temp_base = ws.temp_base - 0.5 * spec.cold
            ws.temperature = ws.temp_base.clone()
        self.we.update_enclosure(ws)
        maxk = max(1, max(len(p) for p in patches))
        pt = torch.full((n_arenas, maxk, 2), -1, dtype=torch.long, device=d)
        for a, pts in enumerate(patches):
            for j, (py, px) in enumerate(pts):
                pt[a, j, 0], pt[a, j, 1] = py, px
        pool = AgentPool(n, self.L, cfg, d)
        gen = torch.Generator(device=d); gen.manual_seed(derive_seed("arena_run", seed) % (2 ** 62))
        arena_of = torch.arange(n_arenas, device=d).repeat_interleave(per_arena)
        sy = c if spec.spawn_side == "center" else c
        sx = c if spec.spawn_side == "center" else 2
        off_rng = np.random.default_rng(derive_seed("arena_spawn", seed) % (2 ** 32))
        offs = torch.as_tensor(off_rng.uniform(-1.2, 1.2, size=(n, 2)), device=d, dtype=torch.float32)
        base = origins[arena_of].float()
        pos = torch.stack([base[:, 1] + sx + 0.5 + offs[:, 0], base[:, 0] + sy + 0.5 + offs[:, 1]], 1)
        if body_cap is None:
            body_cap = self.capability(genomes)
        aids = self.ids.take(n)
        zeros = np.zeros(n, np.int64)
        pool.spawn(torch.arange(n, device=d), genomes, pos, aids, aids, zeros, aids, zeros, 0,
                   torch.full((n,), energy, device=d), body_cap=body_cap, gen=gen)
        pool.body_measured[:] = True
        if W_override is not None:
            pool.W[:n] = W_override
        if x_override is not None:
            pool.x[:n] = x_override
        ledger = StructureLedger(max(256, n * 8), d)
        return Trial(ws=ws, pool=pool, ledger=ledger, gen=gen, spec=spec, n_arenas=n_arenas, per_arena=per_arena,
                     arena_of=arena_of, origins=origins, label=lab, seed=seed, patch_tiles=pt, barrier_x=barrier_x)

    # ------------------------------------------------------------------ execução
    def run_gen(self, trial: Trial, ticks: int, ctrl_fn: Optional[Callable[[int, Trial], Optional[ArenaControl]]] = None,
                record: Sequence[str] = ("energy", "pos"), chunk: int = 16, every: int = 1,
                bodies: Optional[Dict[int, BodyRegistry]] = None, name: str = "") -> Generator[None, None, Dict[str, torch.Tensor]]:
        """Executa `ticks` passos; grava campos por tick. Rende (yield) a cada `chunk` ticks."""
        pool, ws = trial.pool, trial.ws
        rec: Dict[str, List[torch.Tensor]] = {k: [] for k in record}
        prev_gain = pool.stats[:, 0].clone()
        prev_place = pool.stats[:, 2].clone()
        self.current, self.current_name = trial, name
        for t in range(ticks):
            ctrl = ctrl_fn(trial.tick, trial) if ctrl_fn is not None else None
            if ctrl is None:
                ctrl = ArenaControl(no_repro=True)
            ctrl.no_repro = True
            self.engine.step(ws, pool, trial.tick, trial.gen, None, self.ids, trial.ledger, bodies=bodies, ctrl=ctrl,
                             record_interactions=False, hi_override=trial.n)
            if trial.tick % self.cfg.env_every == 0:
                iy, ix = WorldEngine.tile_index(ws, pool.pos[pool.alive])
                ws.tick = trial.tick
                self.we.env_step(ws, trial.gen, torch.stack([iy, ix]))
            if t % every == 0:
                for k in record:
                    if k == "gain":
                        g = pool.stats[:, 0] - prev_gain
                        rec[k].append(g.clone())
                    elif k == "placed":
                        rec[k].append(pool.stats[:, 2] - prev_place)
                    elif k == "alive":
                        rec[k].append(pool.alive.clone())
                    else:
                        rec[k].append(getattr(pool, k).clone())
            prev_gain = pool.stats[:, 0].clone()
            prev_place = pool.stats[:, 2].clone()
            trial.tick += 1
            self.ticks_run += 1
            if (t + 1) % chunk == 0:
                yield
        out = {k: torch.stack(v) if v else torch.zeros(0) for k, v in rec.items()}
        return out

    def run(self, trial: Trial, ticks: int, **kw) -> Dict[str, torch.Tensor]:
        return drive(self.run_gen(trial, ticks, **kw))

    # ------------------------------------------------------------------ métricas de ensaio
    @staticmethod
    def crossed(trial: Trial) -> torch.Tensor:
        if trial.barrier_x < 0:
            return torch.zeros(trial.n, dtype=torch.bool, device=trial.pool.device)
        lx = trial.pool.pos[:, 0] - trial.origins[trial.arena_of, 1].float()
        return lx > trial.barrier_x + 2

    def view(self) -> Optional[Dict[str, Any]]:
        """Visão compacta do ensaio em curso para a 'câmera do cientista' (somente leitura)."""
        tr = self.current
        if tr is None:
            return None
        k = min(tr.n, 512)
        p = tr.pool
        return {"name": self.current_name, "tick": tr.tick, "n_arenas": tr.n_arenas, "size": self.cfg.probe.arena_size,
                "H": tr.ws.H, "W": tr.ws.W, "pos": p.pos[:k].detach().cpu().numpy().astype(np.float32),
                "alive": p.alive[:k].cpu().numpy(), "energy": p.energy[:k].cpu().numpy().astype(np.float32),
                "struct": torch.nonzero(tr.ws.struct_h[: min(tr.ws.H, 256), : min(tr.ws.W, 256)] > 0)[:4000].cpu().numpy(),
                "resource": F.avg_pool2d(tr.ws.resource[None, None], 2)[0, 0].cpu().numpy().astype(np.float16)
                if tr.ws.H >= 2 else None}


def drive(g: Generator):
    """Executa um gerador até o fim e devolve o valor de retorno."""
    try:
        while True:
            next(g)
    except StopIteration as e:
        return e.value
