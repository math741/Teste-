# %% [markdown]
# ## 04 — Tipos centrais: genoma de desenvolvimento, linhagem persistente e pools Struct-of-Arrays na GPU
# * `DevelopmentGenome` (layout vetorial): **regras** que geram corpo e substrato de sinais — nunca o corpo final.
#   Seções: GRN de morfógenos 3D, leitura de propriedades contínuas da matéria (rigidez, contração, adesão,
#   condutividade, sensibilidade…), inervação motora, substrato de sinais (conexões, constantes de tempo, modulação),
#   regra local de plasticidade (forma ABCD evoluível), cronograma de exploração motora, traços de vida, assinatura
#   visual genética e parâmetros de mutação auto-adaptativos.
# * Sensores e ações são **canais físicos** (campos do mundo, contato, sinais emitidos). Nenhum canal recebe objetos
#   classificados, rótulos sociais ou palavras.

# %%
SENSOR_NAMES = ["bias", "resource_here", "resource_grad_fwd", "resource_grad_left", "light", "temperature",
                "slope_fwd", "slope_left", "water_ahead", "structure_here", "enclosure_here", "marks_here",
                "sound_0", "sound_1", "sound_2", "sound_3", "neighbor_density", "neighbor_dir_fwd", "neighbor_dir_left",
                "neighbor_color_0", "neighbor_color_1", "neighbor_motion", "neighbor_carry", "intero_energy",
                "intero_integrity", "intero_fatigue", "intero_damage_rate", "proprio_carry", "contact", "exposure"]
ACTION_NAMES = ["forward", "turn", "emit_0", "emit_1", "emit_2", "emit_3", "consume", "pick_material",
                "place_material", "reproduce", "deposit_mark", "transfer_to_nearest", "take_from_nearest",
                "display_0", "display_1", "rest"]
N_SENSORS, N_ACTIONS = len(SENSOR_NAMES), len(ACTION_NAMES)
N_PROPS = 10
PROP_NAMES = ["stiffness", "damping", "adhesion", "contraction", "conductivity", "sensitivity", "permeability",
              "signal_capacity", "plasticity", "density"]
N_MOTOR = 3                                     # forward, turn, oscilador interno
# Técnicas = padrões ação×contexto detectados de FORA (o agente não recebe esses rótulos)
TECH_NAMES = ["fill_water", "stack_wall", "coop_rich", "give", "mark", "take", "shelter_use", "cross_filled"]
N_TECH = len(TECH_NAMES)


@dataclass
class GeneSpec:
    name: str
    shape: Tuple[int, ...]
    lo: float
    hi: float
    kind: str = "float"                         # float | log
    section: str = "dev"


class GenomeLayout:
    """Layout plano do DevelopmentGenome; mutação, crossover e limites vetorizados em torch (GPU)."""

    def __init__(self, cfg: CivConfig):
        k, N, I, O = cfg.n_morph, cfg.n_units, N_SENSORS, N_ACTIONS
        self.k, self.N, self.I, self.O = k, N, I, O
        lo_age, hi_age = cfg.max_age_range
        specs = [
            GeneSpec("dev_Wm", (k, k), -2, 2), GeneSpec("dev_bm", (k,), -1, 1), GeneSpec("dev_decay", (k,), 0, 0.5),
            GeneSpec("dev_diff", (k,), 0, 0.25), GeneSpec("dev_seed", (k,), -1, 1), GeneSpec("dev_axis", (3,), -1, 1),
            GeneSpec("dev_gw", (k,), -2, 2), GeneSpec("dev_gthr", (), -1, 1), GeneSpec("dev_maxfrac", (), 0.05, 1.0),
            GeneSpec("dev_noise", (), 0.0, 0.3),
            GeneSpec("body_Wp", (N_PROPS, k), -2, 2, section="body"), GeneSpec("body_bp", (N_PROPS,), -2, 2, section="body"),
            GeneSpec("body_innerv", (N_MOTOR, k), -2, 2, section="body"),
            GeneSpec("body_osc", (), 0.5, 4.0, section="body"),
            GeneSpec("ctl_Win", (N, I), -2, 2, section="signal"), GeneSpec("ctl_W0", (N, N), -2, 2, section="signal"),
            GeneSpec("ctl_Wout", (O, N), -2, 2, section="signal"), GeneSpec("ctl_b", (N,), -1, 1, section="signal"),
            GeneSpec("ctl_bout", (O,), -1, 1, section="signal"), GeneSpec("ctl_logtau", (N,), 0.0, 2.5, section="signal"),
            GeneSpec("ctl_wmod", (N,), -2, 2, section="signal"), GeneSpec("ctl_bmod", (), -2, 2, section="signal"),
            GeneSpec("plast_eta", (), 0.0, 0.05, section="plasticity"),
            GeneSpec("plast_abcd", (4,), -1, 1, section="plasticity"),
            GeneSpec("plast_decay", (), 0.0, 0.05, section="plasticity"),
            GeneSpec("babble_sigma", (), 0.0, 0.5, section="behavior"),
            GeneSpec("babble_tau", (), 10.0, 2000.0, "log", section="behavior"),
            GeneSpec("life_maturity", (), 60.0, 600.0, section="life"),
            GeneSpec("life_span", (), float(lo_age), float(hi_age), section="life"),
            GeneSpec("life_repro", (), 0.5, 0.95, section="life"),
            GeneSpec("display", (3,), 0.0, 1.0, section="display"),
            GeneSpec("mut_sigma", (), 0.005, 0.3, "log", section="mutation"),
            GeneSpec("mut_struct", (), 0.0, 0.3, section="mutation"),
        ]
        self.specs = {s.name: s for s in specs}
        self.slices: Dict[str, Tuple[int, int]] = {}
        i = 0
        lo, hi, logm = [], [], []
        for s in specs:
            n = int(np.prod(s.shape)) if s.shape else 1
            self.slices[s.name] = (i, i + n)
            lo += [s.lo] * n; hi += [s.hi] * n; logm += [s.kind == "log"] * n
            i += n
        self.P = i
        self.lo = torch.tensor(lo, dtype=torch.float32)
        self.hi = torch.tensor(hi, dtype=torch.float32)
        self.is_log = torch.tensor(logm)
        self.section_mask = {}
        for s in specs:
            m = torch.zeros(self.P, dtype=torch.bool)
            a, b = self.slices[s.name]
            m[a:b] = True
            self.section_mask[s.section] = self.section_mask.get(s.section, torch.zeros(self.P, dtype=torch.bool)) | m

    def view(self, G: torch.Tensor, name: str) -> torch.Tensor:
        a, b = self.slices[name]
        s = self.specs[name]
        return G[..., a:b].reshape(*G.shape[:-1], *s.shape) if s.shape else G[..., a]

    def to(self, device):
        self.lo, self.hi, self.is_log = self.lo.to(device), self.hi.to(device), self.is_log.to(device)
        self.section_mask = {k: v.to(device) for k, v in self.section_mask.items()}
        return self

    def clamp(self, G: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(torch.nan_to_num(G), self.hi), self.lo)

    def random(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.lo.device
        u = torch.rand(n, self.P, generator=gen, device=dev)
        G = self.lo + u * (self.hi - self.lo)
        logv = torch.exp(torch.log(self.lo.clamp(min=1e-6)) + u * (torch.log(self.hi) - torch.log(self.lo.clamp(min=1e-6))))
        G = torch.where(self.is_log, logv, G)
        # priores mais suaves para conexões (a maioria dos genomas aleatórios é lixo — é para isso que existe seleção)
        for name, scale, dens in [("ctl_Win", 0.7, 0.35), ("ctl_W0", 0.6, 0.25), ("ctl_Wout", 0.8, 0.5)]:
            a, b = self.slices[name]
            w = torch.randn(n, b - a, generator=gen, device=dev) * scale
            G[:, a:b] = w * (torch.rand(n, b - a, generator=gen, device=dev) < dens)
        for name, val in [("mut_sigma", 0.04), ("mut_struct", 0.05), ("plast_decay", 0.005)]:
            G[:, self.slices[name][0]] = val
        return self.clamp(G)

    def mutate(self, G: torch.Tensor, gen: torch.Generator, sigma_mult: float = 1.0) -> torch.Tensor:
        """Mutação vetorizada: σ auto-adaptativo (log-normal), mutação gaussiana pequena em ~12% dos genes, rara mutação
        forte (reamostragem), mutação estrutural (poda/criação de conexões e reamostragem do programa de
        desenvolvimento) com taxa herdada."""
        n, P, dev = G.shape[0], self.P, G.device
        G = G.clone()
        ms_i, st_i = self.slices["mut_sigma"][0], self.slices["mut_struct"][0]
        G[:, ms_i] = G[:, ms_i] * torch.exp(0.2 * torch.randn(n, generator=gen, device=dev))
        G[:, st_i] = G[:, st_i] * torch.exp(0.2 * torch.randn(n, generator=gen, device=dev))
        G = self.clamp(G)
        sig = (G[:, ms_i] * sigma_mult).unsqueeze(1)
        rng_w = (self.hi - self.lo).unsqueeze(0)
        mask = torch.rand(n, P, generator=gen, device=dev) < 0.12
        G = G + mask * torch.randn(n, P, generator=gen, device=dev) * sig * rng_w
        strong = torch.rand(n, P, generator=gen, device=dev) < 0.003
        fresh = self.lo + torch.rand(n, P, generator=gen, device=dev) * (self.hi - self.lo)
        G = torch.where(strong, fresh, G)
        struct_rate = G[:, st_i].unsqueeze(1)
        for name, dens in [("ctl_W0", 0.02), ("ctl_Win", 0.02)]:
            a, b = self.slices[name]
            ev = torch.rand(n, 1, generator=gen, device=dev) < struct_rate
            prune = ev & (torch.rand(n, b - a, generator=gen, device=dev) < dens)
            grow = ev & (torch.rand(n, b - a, generator=gen, device=dev) < dens)
            seg = G[:, a:b]
            seg = torch.where(prune, torch.zeros_like(seg), seg)
            seg = torch.where(grow & (seg == 0), torch.randn(n, b - a, generator=gen, device=dev) * 0.6, seg)
            G[:, a:b] = seg
        devm = self.section_mask["dev"].unsqueeze(0)
        ev = (torch.rand(n, 1, generator=gen, device=dev) < struct_rate * 0.3) & devm
        resample = ev & (torch.rand(n, P, generator=gen, device=dev) < 0.25)
        G = torch.where(resample, fresh, G)
        return self.clamp(G)

    def crossover(self, A: torch.Tensor, B: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Recombinação opcional por seção (blocos funcionais herdados juntos)."""
        out = A.clone()
        for sec, m in self.section_mask.items():
            if sec == "mutation":
                continue
            pick = (torch.rand(A.shape[0], 1, generator=gen, device=A.device) < 0.5) & m.unsqueeze(0)
            out = torch.where(pick, B, out)
        return out

    def distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        rngw = (self.hi - self.lo).clamp(min=1e-6)
        return (((A - B) / rngw) ** 2).mean(-1).sqrt()


class IdCounter:
    def __init__(self, start: int = 1):
        self.value = int(start)

    def take(self, n: int) -> np.ndarray:
        out = np.arange(self.value, self.value + n, dtype=np.int64)
        self.value += n
        return out


class LineageStore:
    """Linhagem de todos os nascimentos (uid, pai, tick, origem, geração). Índice recente em memória (limitado) +
    shards persistentes comprimidos (write-behind) — ancestralidade profunda é lida dos shards sob demanda."""
    ORIGINS = {"founder": 0, "birth": 1, "immigrant_qd": 2, "disaggregated": 3, "imported": 4, "fork": 5}

    def __init__(self, paths: Optional["CivPaths"], recent: int = 300000):
        self.paths = paths
        self.recent_cap = recent
        self.recent: "collections.OrderedDict[int, Tuple[int, int, int, int]]" = collections.OrderedDict()
        self._pending: List[Tuple[int, int, int, int, int]] = []
        self.shards: List[str] = []
        self.total = 0
        self.children_count: Dict[int, int] = {}

    def add(self, uids: np.ndarray, parents: np.ndarray, tick: int, origin: str, gens: np.ndarray):
        oc = self.ORIGINS.get(origin, 4)
        for u, p, g in zip(uids.tolist(), parents.tolist(), gens.tolist()):
            self.recent[u] = (p, tick, oc, g)
            self._pending.append((u, p, tick, oc, g))
            if p > 0:
                self.children_count[p] = self.children_count.get(p, 0) + 1
        self.total += len(uids)
        while len(self.recent) > self.recent_cap:
            self.recent.popitem(last=False)
        if len(self._pending) >= 50000:
            self.flush()
        if len(self.children_count) > 4 * self.recent_cap:
            keep = sorted(self.children_count.items(), key=lambda x: -x[1])[: self.recent_cap]
            self.children_count = dict(keep)

    def flush(self):
        if not self._pending or self.paths is None:
            return
        arr = np.asarray(self._pending, dtype=np.int64)
        name = os.path.join(self.paths.lineages, f"lineage_{len(self.shards):06d}_{int(time.time())}.npz")
        np.savez_compressed(name, data=arr)
        self.shards.append(name)
        self._pending = []

    def parent_of(self, uid: int) -> Optional[int]:
        if uid in self.recent:
            return self.recent[uid][0]
        for p in reversed(self.shards):
            try:
                d = np.load(p)["data"]
            except Exception:
                continue
            hit = np.nonzero(d[:, 0] == uid)[0]
            if hit.size:
                return int(d[hit[0], 1])
        return None

    def ancestors(self, uid: int, depth: int = 50) -> List[int]:
        out, cur = [], uid
        for _ in range(depth):
            p = self.parent_of(cur)
            if p is None or p <= 0:
                break
            out.append(p); cur = p
        return out

    def state_dict(self):
        self.flush()
        tail = list(self.recent.items())[-20000:]
        return {"recent": tail, "shards": self.shards, "total": self.total,
                "children": sorted(self.children_count.items(), key=lambda x: -x[1])[:20000]}

    def load_state_dict(self, s):
        self.recent = collections.OrderedDict((int(k), tuple(v)) for k, v in s["recent"])
        self.shards = [p for p in s["shards"] if os.path.exists(p)]
        self.total = int(s["total"]); self.children_count = {int(k): int(v) for k, v in s.get("children", [])}


class AgentPool:
    """Struct-of-Arrays na GPU, capacidade fixa (sem listas Python no caminho quente). Cada slot guarda o genótipo
    completo do indivíduo; os parâmetros do substrato de sinais são *views* do genoma (sem cópia)."""
    FLOAT_FIELDS = ["heading", "speed", "energy", "integrity", "fatigue", "age", "carried", "last_damage", "mod",
                    "repro_cool", "contact"]

    def __init__(self, cap: int, layout: GenomeLayout, cfg: CivConfig, device: torch.device):
        self.cap, self.layout, self.device = int(cap), layout, device
        N, O = cfg.n_units, N_ACTIONS
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.alive = z(self.cap, dt=torch.bool)
        self.aid = z(self.cap, dt=torch.int64)
        self.guid = z(self.cap, dt=torch.int64)
        self.parent = z(self.cap, dt=torch.int64)
        self.root = z(self.cap, dt=torch.int64)
        self.generation = z(self.cap, dt=torch.int32)
        self.birth = z(self.cap, dt=torch.int64)
        self.species = z(self.cap, dt=torch.int32)
        self.fidelity = z(self.cap, dt=torch.int8)
        self.body_slot = torch.full((self.cap,), -1, dtype=torch.int32, device=device)
        self.pos = z(self.cap, 2)
        for f in self.FLOAT_FIELDS:
            setattr(self, f, z(self.cap))
        self.genome = z(self.cap, layout.P)
        self.x = z(self.cap, N)
        self.W = z(self.cap, N, N)
        self.out = z(self.cap, O)
        self.emit = z(self.cap, cfg.n_sound)
        self.color = z(self.cap, 2)
        self.body_cap = z(self.cap, 6)                 # [velocidade líquida, custo, nós, giro, ângulo da marcha, oscilação]
        self.body_measured = z(self.cap, dt=torch.bool)
        self.stats = z(self.cap, 8)
        self.signature = z(self.cap, O)
        self.culture_type = torch.full((self.cap,), -1, dtype=torch.int32, device=device)
        self.last_partner = z(self.cap, dt=torch.int64)
        self.blocked = z(self.cap)
        self.sense_cache = z(self.cap, N_SENSORS)
        self.intervened = z(self.cap, dt=torch.bool)
        self.tech = z(self.cap, dt=torch.int32)        # bits de técnicas observadas (instrumento externo, invisível ao agente)

    TENSOR_FIELDS = ["alive", "aid", "guid", "parent", "root", "generation", "birth", "species", "fidelity", "body_slot",
                     "pos", "heading", "speed", "energy", "integrity", "fatigue", "age", "carried", "last_damage", "mod",
                     "repro_cool", "genome", "x", "W", "out", "emit", "color", "body_cap", "body_measured", "stats",
                     "signature", "culture_type", "last_partner", "blocked", "sense_cache", "intervened", "contact", "tech"]

    def n_alive(self) -> int:
        return int(self.alive.sum().item())

    def free_slots(self, n: int) -> torch.Tensor:
        free = torch.nonzero(~self.alive).flatten()
        return free[:n]

    def spawn(self, slots: torch.Tensor, genomes: torch.Tensor, pos: torch.Tensor, aids: np.ndarray, guids: np.ndarray,
              parents: np.ndarray, roots: np.ndarray, gens: np.ndarray, tick: int, energy: torch.Tensor,
              body_cap: Optional[torch.Tensor] = None, species: Optional[torch.Tensor] = None,
              gen: Optional[torch.Generator] = None):
        d = self.device
        n = slots.numel()
        if n == 0:
            return
        L = self.layout
        self.alive[slots] = True
        self.aid[slots] = torch.as_tensor(aids, device=d)
        self.guid[slots] = torch.as_tensor(guids, device=d)
        self.parent[slots] = torch.as_tensor(parents, device=d)
        self.root[slots] = torch.as_tensor(roots, device=d)
        self.generation[slots] = torch.as_tensor(gens, device=d, dtype=torch.int32)
        self.birth[slots] = int(tick)
        self.species[slots] = species if species is not None else 0
        self.fidelity[slots] = 1
        self.body_slot[slots] = -1
        self.pos[slots] = pos
        self.heading[slots] = torch.rand(n, generator=gen, device=d) * 2 * math.pi
        for f in ["speed", "fatigue", "age", "carried", "last_damage", "mod", "repro_cool", "blocked", "contact"]:
            getattr(self, f)[slots] = 0.0
        self.energy[slots] = energy
        self.integrity[slots] = 1.0
        self.genome[slots] = genomes
        self.x[slots] = 0.0
        self.W[slots] = L.view(genomes, "ctl_W0")
        self.out[slots] = 0.0
        self.emit[slots] = 0.0
        disp = L.view(genomes, "display")
        self.color[slots] = disp[:, :2]
        if body_cap is not None:
            self.body_cap[slots] = body_cap
            self.body_measured[slots] = False
        else:
            # ainda não medido: sem termo de corpo (só rastejo basal) até o desenvolvimento ser medido na física
            self.body_cap[slots] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=d)
            self.body_measured[slots] = False
        self.stats[slots] = 0.0
        self.signature[slots] = 0.0
        self.culture_type[slots] = -1
        self.last_partner[slots] = 0
        self.intervened[slots] = False
        self.tech[slots] = 0

    def state_dict(self) -> Dict[str, np.ndarray]:
        idx = torch.nonzero(self.alive).flatten()
        out = {"cap": self.cap, "n": int(idx.numel())}
        for f in self.TENSOR_FIELDS:
            out[f] = getattr(self, f)[idx].detach().cpu().numpy()
        return out

    def load_state_dict(self, s: Dict[str, Any]):
        n = int(s["n"])
        if n > self.cap:
            raise ValueError(f"checkpoint tem {n} agentes > capacidade {self.cap}")
        self.alive[:] = False
        idx = torch.arange(n, device=self.device)
        for f in self.TENSOR_FIELDS:
            t = getattr(self, f)
            t[idx] = torch.as_tensor(s[f], device=self.device, dtype=t.dtype)
