# %% [markdown]
# ## 05 — WORLD ENGINE: planeta 2.5D/3D multiescala
# Representação híbrida (cada fenômeno com a sua): mapa de tiles em tensores (relevo, água, rios, clima, fertilidade,
# recurso abstrato, depósitos de matéria, luz, tempestades, campos de som/marcas, tráfego), pilhas esparsas de matéria
# construída (módulo 06), agentes como partículas (módulo 10) e corpos como grafos de Matter Nodes (módulos 07–08).
# * Relevo por fBm na GPU; nível do mar por quantil; rios por acumulação de fluxo D8 (rios maiores viram água).
# * Clima simplificado: latitude, altitude, estações, ciclo dia/noite, tempestades advectadas pelo vento.
# * **Event-driven**: chunks sem agentes e sem tempestade atualizam com período maior (dt acumulado, mesma física).
# Tempo artificial: `tick` (unidade da simulação); "dia" e "estação" são ciclos artificiais, não calendário humano.

# %%
@dataclass
class WorldState:
    H: int
    W: int
    height: torch.Tensor
    is_water: torch.Tensor
    filled: torch.Tensor
    river: torch.Tensor
    temp_base: torch.Tensor
    temperature: torch.Tensor
    moisture: torch.Tensor
    fertility: torch.Tensor
    deposit: torch.Tensor
    resource: torch.Tensor
    rich: torch.Tensor
    light: torch.Tensor
    storm: torch.Tensor
    sound: torch.Tensor
    marks: torch.Tensor
    traffic: torch.Tensor
    struct_h: torch.Tensor
    struct_integrity: torch.Tensor
    enclosure: torch.Tensor
    stored: torch.Tensor
    chunk_last: torch.Tensor
    chunk_active: torch.Tensor
    wind: torch.Tensor
    tick: int = 0
    seed: int = 0
    kind: str = "planet"
    arena: Optional[Dict[str, Any]] = None
    struct_dirty: bool = True
    regen_mult: Optional[torch.Tensor] = None
    flags: Dict[str, Any] = field(default_factory=dict)

    TENSORS = ["height", "is_water", "filled", "river", "temp_base", "temperature", "moisture", "fertility", "deposit",
               "resource", "rich", "light", "storm", "sound", "marks", "traffic", "struct_h", "struct_integrity",
               "enclosure", "stored", "chunk_last", "chunk_active", "wind"]

    @property
    def device(self):
        return self.height.device

    def walkable(self) -> torch.Tensor:
        return ((~self.is_water) | self.filled) & (self.struct_h <= 1)

    def state_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k).detach().cpu().numpy() for k in self.TENSORS}
        out.update({"H": self.H, "W": self.W, "tick": self.tick, "seed": self.seed, "kind": self.kind,
                    "arena": self.arena, "flags": self.flags})
        return out

    @staticmethod
    def from_state_dict(s: Dict[str, Any], device: torch.device) -> "WorldState":
        t = {k: torch.as_tensor(s[k], device=device) for k in WorldState.TENSORS}
        ws = WorldState(H=int(s["H"]), W=int(s["W"]), tick=int(s["tick"]), seed=int(s["seed"]), kind=s.get("kind", "planet"),
                        arena=s.get("arena"), flags=dict(s.get("flags", {})), **t)
        ws.struct_dirty = True
        return ws

    def clone(self) -> "WorldState":
        t = {k: getattr(self, k).clone() for k in self.TENSORS}
        return WorldState(H=self.H, W=self.W, tick=self.tick, seed=self.seed, kind=self.kind,
                          arena=copy.deepcopy(self.arena), flags=dict(self.flags), **t)


def fbm(H: int, W: int, octaves: int, gen: torch.Generator, device, base: int = 4, persistence: float = 0.5) -> torch.Tensor:
    out = torch.zeros(H, W, device=device)
    amp, res, tot = 1.0, base, 0.0
    for _ in range(octaves):
        g = torch.rand(1, 1, res + 1, res + 1, generator=gen, device=device)
        out += amp * F.interpolate(g, size=(H, W), mode="bilinear", align_corners=True)[0, 0]
        tot += amp; amp *= persistence; res *= 2
    out = out / tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def flow_accumulation(height: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Acumulação de fluxo D8 (descida mais íngreme), processando do mais alto para o mais baixo."""
    H, W = height.shape
    pad = np.pad(height, 1, mode="edge")
    best = np.full((H, W), np.inf); down = np.full((H, W), -1, dtype=np.int64)
    idx = np.arange(H * W).reshape(H, W)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            yy = np.clip(np.arange(H)[:, None] + dy, 0, H - 1); xx = np.clip(np.arange(W)[None, :] + dx, 0, W - 1)
            better = (nb < best) & (nb < height)
            best = np.where(better, nb, best)
            down = np.where(better, idx[yy, xx], down)
    acc = np.ones(H * W)
    order = np.argsort(-height.ravel())
    d = down.ravel(); wflat = water.ravel()
    for i in order:
        j = d[i]
        if j >= 0 and not wflat[i]:
            acc[j] += acc[i]
    return acc.reshape(H, W)


def blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if x.dim() == 2:
        return F.avg_pool2d(x[None, None], k, 1, k // 2, count_include_pad=False)[0, 0]
    return F.avg_pool2d(x[None], k, 1, k // 2, count_include_pad=False)[0]


class WorldEngine:
    """Gera e atualiza mundos (planeta principal, forks e arenas de laboratório) com a mesma física."""

    def __init__(self, cfg: CivConfig, device: torch.device):
        self.cfg, self.device = cfg, device
        self.R_MAX = 1.0
        self.regen_rate = 0.004
        self.sound_decay = 0.55
        self.mark_decay = 0.999

    # ------------------------------------------------------------------ criação
    def _empty(self, H: int, W: int, seed: int, kind: str) -> WorldState:
        d, cfg = self.device, self.cfg
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=d)
        Hc, Wc = max(1, math.ceil(H / cfg.chunk)), max(1, math.ceil(W / cfg.chunk))
        return WorldState(H=H, W=W, height=z(H, W), is_water=z(H, W, dt=torch.bool), filled=z(H, W, dt=torch.bool),
                          river=z(H, W), temp_base=z(H, W), temperature=z(H, W), moisture=z(H, W), fertility=z(H, W),
                          deposit=z(H, W), resource=z(H, W), rich=z(H, W, dt=torch.bool), light=z(H, W) + 1.0,
                          storm=z(H, W), sound=z(cfg.n_sound, H, W), marks=z(H, W), traffic=z(H, W),
                          struct_h=z(H, W, dt=torch.int32), struct_integrity=z(H, W), enclosure=z(H, W), stored=z(H, W),
                          chunk_last=z(Hc, Wc, dt=torch.int64), chunk_active=z(Hc, Wc, dt=torch.bool),
                          wind=torch.tensor([0.6, 0.2], device=d), seed=seed, kind=kind)

    def generate(self, seed: int, H: Optional[int] = None, W: Optional[int] = None) -> WorldState:
        cfg, d = self.cfg, self.device
        H, W = H or cfg.world_h, W or cfg.world_w
        g = torch.Generator(device=d); g.manual_seed(int(seed))
        ws = self._empty(H, W, seed, "planet")
        base = fbm(H, W, 6, g, d, base=3)
        ridge = 1 - (2 * fbm(H, W, 5, g, d, base=4) - 1).abs()
        yy = torch.linspace(-1, 1, H, device=d).view(H, 1); xx = torch.linspace(-1, 1, W, device=d).view(1, W)
        falloff = (1 - 0.55 * (xx ** 2 + yy ** 2)).clamp(0, 1)
        h = (0.65 * base + 0.35 * ridge ** 2) * (0.55 + 0.45 * falloff)
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)
        sea = torch.quantile(h.flatten()[:: max(1, (H * W) // 200000)], 0.36)
        water = h < sea
        acc = flow_accumulation(h.cpu().numpy(), water.cpu().numpy())
        river = torch.as_tensor(np.log1p(acc) / max(1e-9, np.log1p(acc).max()), device=d, dtype=torch.float32)
        river = river * (~water)
        big_river = river > torch.quantile(river[~water].flatten()[:: max(1, (H * W) // 200000)], 0.985)
        water = water | big_river
        ws.height = h * cfg.height_scale
        ws.is_water = water
        ws.river = river
        lat = yy.abs().expand(H, W)
        land_h = ((h - sea).clamp(min=0) / (1 - sea + 1e-9))
        ws.temp_base = (1.0 - 0.75 * lat - 0.55 * land_h).clamp(0, 1)
        wetsrc = water.float() + 0.6 * river
        moist = wetsrc
        for _ in range(4):
            moist = blur(moist, 7)
        ws.moisture = (moist / (moist.max() + 1e-9) * 0.8 + 0.2 * fbm(H, W, 4, g, d, base=6)).clamp(0, 1)
        fert = (ws.moisture * (1 - (ws.temp_base - 0.62).abs() * 1.4)).clamp(0, 1) * (~water)
        ws.fertility = fert
        gy, gx = torch.gradient(h)
        slope = torch.sqrt(gy ** 2 + gx ** 2)
        rock = (slope / (slope.max() + 1e-9) * 1.2 + land_h * 0.8).clamp(0, 1)
        ws.deposit = (rock * 8.0).floor() * (~water)
        ws.resource = fert * self.R_MAX
        n_rich = max(2, (H * W) // 900)
        cy = torch.randint(0, H, (n_rich,), generator=g, device=d); cx = torch.randint(0, W, (n_rich,), generator=g, device=d)
        rich = torch.zeros(H, W, dtype=torch.bool, device=d)
        Y, X = torch.meshgrid(torch.arange(H, device=d), torch.arange(W, device=d), indexing="ij")
        for i in range(n_rich):
            rich |= ((Y - cy[i]) ** 2 + (X - cx[i]) ** 2) <= 4
        ws.rich = rich & (fert > 0.25)
        self.update_light(ws)
        ws.temperature = ws.temp_base.clone()
        ws.flags["sea_level"] = float(sea * cfg.height_scale)
        return ws

    def make_arena_world(self, n_arenas: int, arena: int, seed: int, base_resource: float = 0.0,
                         day_night: bool = False, weather: bool = False) -> WorldState:
        """Arenas de laboratório: sub-mundos idênticos separados por paredes altas, num único mundo (vetorização de
        ensaios com a mesma física do planeta)."""
        side = int(math.ceil(math.sqrt(n_arenas)))
        cell = arena + 2
        H = W = side * cell
        ws = self._empty(H, W, seed, "arena")
        ws.height[:] = 2.0
        ws.temp_base[:] = 0.6; ws.temperature[:] = 0.6; ws.moisture[:] = 0.5
        ws.fertility[:] = 0.0; ws.resource[:] = base_resource
        wall = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        for i in range(side + 1):
            a = min(H - 1, i * cell)
            wall[a, :] = True; wall[:, a] = True
            if a + cell - 1 < H:
                wall[a + cell - 1, :] = True; wall[:, a + cell - 1] = True
        ws.struct_h = torch.where(wall, torch.full_like(ws.struct_h, self.cfg.max_stack), ws.struct_h)
        ws.struct_integrity = wall.float() * 1e9
        origins = [(1 + (i // side) * cell, 1 + (i % side) * cell) for i in range(n_arenas)]
        ws.arena = {"n": n_arenas, "size": arena, "cell": cell, "side": side, "origins": origins}
        ws.flags.update({"day_night": day_night, "weather": weather})
        ws.struct_dirty = True
        return ws

    # ------------------------------------------------------------------ amostragem (sensores físicos)
    @staticmethod
    def tile_index(ws: WorldState, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = pos[:, 0].floor().long().clamp(0, ws.W - 1)
        iy = pos[:, 1].floor().long().clamp(0, ws.H - 1)
        return iy, ix

    @staticmethod
    def sample(field_: torch.Tensor, iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        if field_.dim() == 3:
            return field_[:, iy, ix].T
        return field_[iy, ix]

    @staticmethod
    def gradient_local(field_: torch.Tensor, iy, ix, heading: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = field_.shape
        gx = field_[iy, (ix + 1).clamp(max=W - 1)] - field_[iy, (ix - 1).clamp(min=0)]
        gy = field_[(iy + 1).clamp(max=H - 1), ix] - field_[(iy - 1).clamp(min=0), ix]
        c, s = torch.cos(heading), torch.sin(heading)
        return gx * c + gy * s, -gx * s + gy * c

    # ------------------------------------------------------------------ dinâmica ambiental
    def update_light(self, ws: WorldState):
        if ws.kind == "arena" and not ws.flags.get("day_night", False):
            ws.light[:] = 1.0
            return
        phase = 2 * math.pi * (ws.tick / self.cfg.day_period)
        xx = torch.linspace(0, 2 * math.pi, ws.W, device=self.device).view(1, ws.W)
        lat = torch.linspace(-1, 1, ws.H, device=self.device).view(ws.H, 1).abs()
        sun = torch.cos(xx - phase) * torch.cos(lat * 1.1)
        ws.light = (sun.clamp(min=0) * (1 - 0.6 * ws.storm)).clamp(0, 1)

    def env_step(self, ws: WorldState, gen: torch.Generator, agent_tiles: Optional[torch.Tensor] = None):
        """Atualização de baixa frequência (env_every ticks). Chunks inativos acumulam dt (event-driven)."""
        cfg, d = self.cfg, self.device
        H, W = ws.H, ws.W
        ch = cfg.chunk
        Hc, Wc = ws.chunk_last.shape
        act = torch.zeros(Hc * Wc, dtype=torch.bool, device=d)
        if agent_tiles is not None and agent_tiles.numel():
            cid = (agent_tiles[0] // ch) * Wc + (agent_tiles[1] // ch)
            act[cid] = True
        storm_c = F.adaptive_max_pool2d(ws.storm[None, None], (Hc, Wc))[0, 0].flatten() > 0.05
        act = (act | storm_c).view(Hc, Wc)
        ws.chunk_active = act
        period = torch.where(act, torch.full_like(ws.chunk_last, cfg.env_every),
                             torch.full_like(ws.chunk_last, cfg.env_every * 4))
        due = (ws.tick - ws.chunk_last) >= period
        dt_c = (ws.tick - ws.chunk_last).float() * due
        ws.chunk_last = torch.where(due, torch.full_like(ws.chunk_last, ws.tick), ws.chunk_last)
        dt = dt_c.repeat_interleave(ch, 0).repeat_interleave(ch, 1)[:H, :W]
        if ws.kind == "planet" or ws.flags.get("weather", False):
            self._weather(ws, gen)
        self.update_light(ws)
        season = math.sin(2 * math.pi * ws.tick / max(1, cfg.season_period))
        hemi = torch.linspace(1, -1, H, device=d).view(H, 1)
        night_cool = 0.12 * (ws.light - 0.5)
        ws.temperature = (ws.temp_base + 0.15 * season * hemi + night_cool - 0.1 * ws.storm).clamp(0, 1)
        regen = self.regen_rate * ws.fertility * (1 + 1.5 * ws.storm) * (1 - ws.resource / self.R_MAX)
        if ws.regen_mult is not None:
            regen = regen * ws.regen_mult
        if ws.kind == "arena" and not ws.flags.get("regen", False):
            regen = regen * 0
        ws.resource = (ws.resource + regen * dt).clamp(0, self.R_MAX)
        ws.marks = ws.marks * (self.mark_decay ** dt)
        ws.traffic = ws.traffic * (0.9995 ** dt)
        weathering = 0.0004 * ws.storm * dt * (ws.struct_h > 0)
        if ws.kind == "planet":
            ws.struct_integrity = ws.struct_integrity - weathering
            collapse = (ws.struct_h > 0) & (ws.struct_integrity <= 0)
            if bool(collapse.any()):
                ws.struct_h = torch.where(collapse, ws.struct_h - 1, ws.struct_h)
                ws.deposit = ws.deposit + collapse.float()            # matéria conservada: bloco vira entulho do tile
                ws.struct_integrity = torch.where(collapse & (ws.struct_h > 0), torch.ones_like(ws.struct_integrity),
                                                  ws.struct_integrity.clamp(min=0))
                ws.struct_dirty = True
        if ws.struct_dirty:
            self.update_enclosure(ws)

    def _weather(self, ws: WorldState, gen: torch.Generator):
        d = self.device
        ws.wind = (ws.wind + 0.05 * torch.randn(2, generator=gen, device=d)).clamp(-1.2, 1.2)
        sx, sy = int(round(float(ws.wind[0]))), int(round(float(ws.wind[1])))
        ws.storm = torch.roll(ws.storm, shifts=(sy, sx), dims=(0, 1)) * 0.985
        if float(torch.rand(1, generator=gen, device=d)) < 0.08 * self.cfg.weather_strength:
            cy = int(torch.randint(0, ws.H, (1,), generator=gen, device=d)); cx = int(torch.randint(0, ws.W, (1,), generator=gen, device=d))
            r = max(3.0, ws.W / 18)
            Y, X = torch.meshgrid(torch.arange(ws.H, device=d), torch.arange(ws.W, device=d), indexing="ij")
            ws.storm = (ws.storm + torch.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * r * r))).clamp(0, 1)

    def update_enclosure(self, ws: WorldState):
        """Abrigo funcional: tile livre cercado por paredes (pilhas ≥2) reduz exposição a tempestade e frio."""
        walls = (ws.struct_h >= 2).float()[None, None]
        k = torch.ones(1, 1, 3, 3, device=self.device); k[0, 0, 1, 1] = 0
        cnt = F.conv2d(F.pad(walls, (1, 1, 1, 1)), k)[0, 0]
        ws.enclosure = ((cnt - 3) / 4).clamp(0, 1) * (ws.struct_h <= 1)
        ws.struct_dirty = False

    def sound_step(self, ws: WorldState):
        ws.sound = blur(ws.sound, 3) * self.sound_decay
