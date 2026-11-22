# %% [markdown]
# ## 4. Física do universo artificial
# $$U_{t+\Delta t} = F(U_t, N_t, G_t, R_t, \Theta)$$
#
# * `U` — campo `[B, C, S, S]` (B universos em paralelo na GPU). Nenhum canal tem significado.
# * `N_t` — vizinhança via kernels evoluídos (anel, gaussiano, Moore, von Neumann, anisotrópico, multiescala) por FFT,
#   mais operadores fixos de gradiente/laplaciano (difusão, transporte).
# * `G_t` — topologia: conexões distantes esparsas cujo peso é um **campo local plástico** (surgem/somem/fortalecem/enfraquecem).
# * `R_t` — recurso abstrato limitado, regenerável, difusível, consumido por crescimento positivo.
# * `Θ` — leis do genoma: interação linear, não linearidades mistas (tanh / sino / ímpar saturante), reação quadrática,
#   difusão, advecção, inércia (ondas), atraso, histerese (traço lento), dissipação, conservação parcial, saturação,
#   constantes temporais por canal, atualização assíncrona parcial, ruído intrínseco.
#
# **Discretização numérica externa ≠ relógio físico interno.** O passo do backend é discreto; cada canal tem sua
# própria taxa (`time_scale/tau`), há atraso e assincronia. O *dt adaptativo* (redução do passo quando |ΔU| excede o limite)
# é uma salvaguarda numérica registrada, não física emergente.
#
# Bordas: `periodic`, `reflective`, `absorbing` (por universo, misturáveis no mesmo batch). A arquitetura (vizinhança por
# kernels + links indexados) foi desenhada para estender a 1D/3D/grafos trocando apenas `pad`/FFT e a indexação dos links.

# %%
@dataclass
class Environment:
    env_id: str
    split: str             # evolution | validation | holdout
    init_kind: str         # blobs | noise | sparse | gradient | center | uniform
    density: float
    amplitude: float
    resource_scale: float
    ext_noise: float
    shock_period: int
    shock_amp: float
    seed: int

    def to_dict(self):
        return asdict(self)


def make_environment_suite(master_seed: int) -> List[Environment]:
    """Ensemble de ambientes separado em evolução / validação / holdout final (nunca visto pela seleção)."""
    rng = np.random.default_rng(derive_seed(master_seed, "envs"))
    spec = [
        ("ENV-A", "evolution", "blobs", 0.5, 0.8, 1.0, 0.000, 0, 0.0),
        ("ENV-B", "evolution", "noise", 1.0, 0.5, 1.0, 0.002, 0, 0.0),
        ("ENV-C", "evolution", "sparse", 0.05, 1.0, 0.8, 0.000, 64, 0.05),
        ("ENV-D", "evolution", "gradient", 0.5, 0.6, 1.2, 0.001, 0, 0.0),
        ("ENV-E", "validation", "blobs", 0.3, 1.0, 0.9, 0.002, 96, 0.03),
        ("ENV-F", "validation", "center", 0.5, 0.9, 1.0, 0.001, 0, 0.0),
        ("ENV-G", "holdout", "noise", 0.7, 0.7, 1.1, 0.003, 80, 0.04),
        ("ENV-H", "holdout", "sparse", 0.10, 0.8, 0.7, 0.001, 0, 0.0),
    ]
    return [Environment(n, sp, k, d, a, r, en, spp, sa, int(rng.integers(1, 2 ** 31 - 1)))
            for n, sp, k, d, a, r, en, spp, sa in spec]


def make_initial_field(env: Environment, C: int, S: int, seed: int, ch_active: np.ndarray) -> np.ndarray:
    """Condição inicial reprodutível por universo (semente própria)."""
    rng = np.random.default_rng(seed)
    U = np.zeros((C, S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    a = env.amplitude
    if env.init_kind == "blobs":
        nb = max(1, int(round(env.density * S * S / 256)))
        for _ in range(nb):
            cy, cx = rng.uniform(0, S, 2); r = rng.uniform(2, max(3, S / 8))
            d2 = np.minimum(np.abs(yy - cy), S - np.abs(yy - cy)) ** 2 + np.minimum(np.abs(xx - cx), S - np.abs(xx - cx)) ** 2
            blob = np.exp(-d2 / (2 * r * r))
            U += blob[None] * rng.normal(0, a, size=(C, 1, 1)).astype(np.float32)
    elif env.init_kind == "noise":
        U = rng.uniform(-a, a, size=(C, S, S)).astype(np.float32) * env.density
    elif env.init_kind == "sparse":
        m = rng.random((C, S, S)) < env.density
        U = np.where(m, rng.choice([-a, a], size=(C, S, S)), 0).astype(np.float32)
    elif env.init_kind == "gradient":
        g = (xx / S - 0.5) * 2 * a
        U = g[None] * rng.normal(1, 0.3, size=(C, 1, 1)) + rng.normal(0, 0.05 * a, size=(C, S, S))
    elif env.init_kind == "center":
        d2 = (yy - S / 2) ** 2 + (xx - S / 2) ** 2
        U = np.exp(-d2 / (2 * (S / 8) ** 2))[None] * rng.normal(0, a, size=(C, 1, 1)) + rng.normal(0, 0.02, size=(C, S, S))
    elif env.init_kind == "uniform":
        U = np.full((C, S, S), a, np.float32) * (rng.choice([-1, 1], size=(C, 1, 1)))
    else:
        raise ValueError(env.init_kind)
    return (U * np.asarray(ch_active, np.float32)[:, None, None]).astype(np.float32)


_KERNEL_CACHE: Dict[tuple, np.ndarray] = {}


def build_kernel_np(shape: int, radius: float, mu: float, sigma: float, aniso: float, angle: float, R: int) -> np.ndarray:
    key = (int(shape), round(float(radius), 3), round(float(mu), 3), round(float(sigma), 3),
           round(float(aniso), 3), round(float(angle), 3), R)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    r = max(float(radius), 1.0)
    if shape == 2:      # Moore
        ri = max(1, int(round(r)))
        k = ((np.abs(yy) <= ri) & (np.abs(xx) <= ri)).astype(np.float64); k[R, R] = 0
    elif shape == 3:    # von Neumann
        ri = max(1, int(round(r)))
        k = ((np.abs(yy) + np.abs(xx)) <= ri).astype(np.float64); k[R, R] = 0
    else:
        if shape == 4:
            c, s = math.cos(angle), math.sin(angle)
            u = c * xx + s * yy; v = (-s * xx + c * yy) / max(aniso, 0.2)
            d = np.sqrt(u * u + v * v) / r
        else:
            d = np.sqrt(xx * xx + yy * yy) / r
        if shape == 1:
            k = np.exp(-0.5 * (d / max(sigma, 0.05)) ** 2)
        else:
            k = np.exp(-0.5 * ((d - mu) / max(sigma, 0.05)) ** 2)
        k[d > 1.0] = 0.0
    s = k.sum()
    if s <= 1e-9:
        k = np.zeros_like(k); k[R - 1, R] = k[R + 1, R] = k[R, R - 1] = k[R, R + 1] = 1.0; s = 4.0
    k = (k / s).astype(np.float32)
    if len(_KERNEL_CACHE) < 50000:
        _KERNEL_CACHE[key] = k
    return k


class GenomeBatch:
    """Tensores das leis de B universos (genomas podem se repetir no batch)."""
    TENSOR_NAMES = ["chm", "W", "m0", "m1", "m2", "g_mu", "g_sig", "g_bias", "g_gain", "diff", "diss", "Q",
                    "vy", "vx", "inertia", "rate", "delay_k", "delay_w", "hyst_rate", "hyst_w", "conserve", "sat",
                    "link_coup", "link_on", "link_oy", "link_ox", "link_w0", "link_eta", "link_decay",
                    "res_on", "res_rho", "res_cons", "res_khalf", "res_diff", "max_rate", "noise", "async_p",
                    "bmode", "Khat"]

    def __init__(self, sim: "Simulator", genomes: List[Genome], S: int, _empty: bool = False):
        self.sim, self.S = sim, S
        cfg = sim.cfg
        self.C, self.K, self.L, self.D = cfg.channels_max, cfg.kernel_slots, cfg.links_max, cfg.delay_max
        self.P = cfg.radius_max
        self.Hp = S + 2 * self.P
        self.genome_ids = [g.genome_id for g in genomes]
        if _empty:
            return
        dev = sim.device
        B = len(genomes)
        C, K, L, R = self.C, self.K, self.L, cfg.radius_max

        def st(name, dtype=np.float32):
            return np.stack([np.asarray(g.params[name]) for g in genomes]).astype(dtype)

        def T(a, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dtype)

        chm = st("ch_active")
        kact = st("k_active")
        self.chm = T(chm).view(B, C, 1, 1)
        W = st("W") * chm[:, :, None, None] * chm[:, None, None, :] * kact[:, None, :, None]
        self.W = T(W)
        gm = st("g_mix")
        self.m0, self.m1, self.m2 = (T(gm[:, :, i]).view(B, C, 1, 1) for i in range(3))
        self.g_mu = T(st("g_mu")).view(B, C, 1, 1)
        self.g_sig = T(st("g_sigma")).view(B, C, 1, 1)
        self.g_bias = T(st("g_bias")).view(B, C, 1, 1)
        self.g_gain = T(st("g_gain")).view(B, C, 1, 1)
        rate = st("time_scale")[:, None] / st("tau")
        self.rate = T(rate).view(B, C, 1, 1)
        diff = np.minimum(st("diff"), 0.2 / np.maximum(rate, 1e-6))         # estabilidade explícita
        self.diff = T(diff).view(B, C, 1, 1)
        self.diss = T(st("diss")).view(B, C, 1, 1)
        self.Q = T(st("Q") * chm[:, :, None] * chm[:, None, :])
        vel = st("vel"); vmax = 0.5 / np.maximum(rate, 1e-6)
        self.vy = T(np.clip(vel[:, :, 0], -vmax, vmax)).view(B, C, 1, 1)
        self.vx = T(np.clip(vel[:, :, 1], -vmax, vmax)).view(B, C, 1, 1)
        self.inertia = T(st("inertia")).view(B, C, 1, 1)
        self.delay_k = T(st("delay_idx", np.int64), torch.long).view(B, C)
        self.delay_w = T(st("delay_w")).view(B, C, 1, 1)
        self.hyst_rate = T(st("hyst_rate")).view(B, C, 1, 1)
        self.hyst_w = T(st("hyst_w")).view(B, C, 1, 1)
        self.conserve = T(st("conserve")).view(B, C, 1, 1)
        self.sat = T(st("sat")).view(B, C, 1, 1)
        self.link_coup = T(st("link_coup")).view(B, C, 1, 1)
        lon = st("link_active")
        self.link_on = T(lon).view(B, L)
        off = st("link_offset")
        oy = np.rint(off[:, :, 0] * S).astype(np.int64); ox = np.rint(off[:, :, 1] * S).astype(np.int64)
        zero = (oy == 0) & (ox == 0); ox = np.where(zero, 1, ox)
        self.link_oy, self.link_ox = T(oy, torch.long), T(ox, torch.long)
        self.link_w0 = T(st("link_weight") * lon).view(B, L, 1, 1)
        self.link_eta = T(st("link_eta")).view(B, 1, 1, 1)
        self.link_decay = T(st("link_decay")).view(B, 1, 1, 1)
        self.res_on = T(st("res_enabled")).view(B, 1, 1, 1)
        self.res_rho = T(st("res_rho")).view(B, 1, 1, 1)
        self.res_cons = T(st("res_cons")).view(B, 1, 1, 1)
        self.res_khalf = T(st("res_khalf")).view(B, 1, 1, 1)
        self.res_diff = T(st("res_diff")).view(B, 1, 1, 1)
        self.max_rate = T(st("max_rate")).view(B, 1, 1, 1)
        self.noise = T(st("noise")).view(B, 1, 1, 1)
        self.async_p = T(st("async_p")).view(B, 1, 1, 1)
        self.bmode = T(st("boundary", np.int64), torch.long).view(B)
        # kernels evoluídos (FFT sobre a grade com padding)
        D = 2 * R + 1
        ks = np.zeros((B, K, D, D), np.float32)
        for b, g in enumerate(genomes):
            p = g.params
            for k in range(K):
                if p["k_active"][k]:
                    ks[b, k] = build_kernel_np(int(p["k_shape"][k]), float(p["k_radius"][k]), float(p["k_mu"][k]),
                                               float(p["k_sigma"][k]), float(p["k_aniso"][k]),
                                               float(p["k_angle"][k]), R)
        kt = T(ks)
        img = F.pad(kt, (0, self.Hp - D, 0, self.Hp - D))
        img = torch.roll(img, shifts=(-R, -R), dims=(2, 3))
        kh = torch.fft.rfft2(img)
        self.Khat = torch.cat([kh, sim.fixed_hat(self.Hp).unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
        self._flags()

    def _flags(self):
        self.B = int(self.chm.shape[0])
        self.has_links = bool((self.link_on > 0).any().item())
        self.link_any = (self.link_on > 0).any(0).tolist()
        self.has_res = bool((self.res_on > 0).any().item())
        self.has_async = bool((self.async_p < 0.999).any().item())
        self.has_noise = bool((self.noise > 0).any().item())
        self.has_delay = self.D > 0 and bool((self.delay_w.abs() > 0).any().item())
        self.has_hyst = bool((self.hyst_w.abs() > 0).any().item())
        self.has_inertia = bool((self.inertia > 0).any().item())
        self.has_plastic = self.has_links and bool((self.link_eta > 0).any().item())
        self.modes_present = sorted(set(self.bmode.tolist()))

    def select(self, idx) -> "GenomeBatch":
        idx_t = torch.as_tensor(idx, device=self.sim.device, dtype=torch.long)
        nb = GenomeBatch(self.sim, [], self.S, _empty=True)
        nb.genome_ids = [self.genome_ids[i] for i in (idx.tolist() if hasattr(idx, "tolist") else idx)]
        for n in self.TENSOR_NAMES:
            setattr(nb, n, getattr(self, n).index_select(0, idx_t))
        nb._flags()
        return nb


@dataclass
class SimState:
    U: torch.Tensor
    Uprev: torch.Tensor
    H: torch.Tensor
    Dbuf: Optional[torch.Tensor]
    ptr: int
    R: torch.Tensor
    R0: torch.Tensor
    G: Optional[torch.Tensor]
    t: int
    bad: torch.Tensor
    group_idx: torch.Tensor
    n_groups: int
    gen: torch.Generator
    ext_noise: torch.Tensor
    shock_period: torch.Tensor
    shock_amp: torch.Tensor
    dt_scale_sum: torch.Tensor
    seeds: List[int]

    def select(self, idx, groups: Optional[Sequence[int]] = None, seed: int = 0) -> "SimState":
        dev = self.U.device
        it = torch.as_tensor(idx, device=dev, dtype=torch.long)
        sel = lambda x: None if x is None else x.index_select(0, it).clone()
        if groups is None:
            groups = list(range(len(it)))
        g = torch.Generator(device=dev); g.manual_seed(int(seed))
        Dbuf = None if self.Dbuf is None else self.Dbuf.index_select(1, it).clone()
        idx_list = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        return SimState(U=sel(self.U), Uprev=sel(self.Uprev), H=sel(self.H), Dbuf=Dbuf, ptr=self.ptr,
                        R=sel(self.R), R0=sel(self.R0), G=sel(self.G), t=self.t, bad=sel(self.bad),
                        group_idx=torch.as_tensor(groups, device=dev, dtype=torch.long),
                        n_groups=int(max(groups)) + 1, gen=g, ext_noise=sel(self.ext_noise),
                        shock_period=sel(self.shock_period), shock_amp=sel(self.shock_amp),
                        dt_scale_sum=sel(self.dt_scale_sum), seeds=[self.seeds[i] for i in idx_list])

    def damage_cells(self, mask: torch.Tensor):
        """Zera todo o estado local (inclusive histórico e traço) nas células marcadas. mask: [B,1,S,S] bool."""
        keep = (~mask).float()
        self.U = self.U * keep; self.Uprev = self.Uprev * keep; self.H = self.H * keep
        if self.Dbuf is not None:
            self.Dbuf = self.Dbuf * keep.unsqueeze(0)


class Simulator:
    """Motor numérico vetorizado. Externo ao universo."""

    def __init__(self, cfg: GenesisConfig, device: torch.device):
        self.cfg, self.device = cfg, device
        self._fixed_cache: Dict[int, torch.Tensor] = {}
        self._grid_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.autocast_dtype = None
        if cfg.precision == "bf16" and device.type == "cuda":
            self.autocast_dtype = torch.bfloat16
        elif cfg.precision == "fp16" and device.type == "cuda":
            self.autocast_dtype = torch.float16
        self.steps_done = 0
        self._physics = self._physics_eager
        if cfg.use_compile:
            try:
                self._physics = torch.compile(self._physics_eager, dynamic=False)
            except Exception:
                self._physics = self._physics_eager

    # ------------------------------------------------------------------
    def fixed_hat(self, Hp: int) -> torch.Tensor:
        """Operadores fixos: laplaciano (5 pontos), gradiente y, gradiente x (diferença central)."""
        if Hp not in self._fixed_cache:
            img = torch.zeros(3, Hp, Hp, device=self.device)
            img[0, 0, 0] = -4; img[0, 1, 0] = 1; img[0, -1, 0] = 1; img[0, 0, 1] = 1; img[0, 0, -1] = 1
            img[1, -1, 0] = 0.5; img[1, 1, 0] = -0.5      # (U[y+1]-U[y-1])/2 como convolução
            img[2, 0, -1] = 0.5; img[2, 0, 1] = -0.5
            self._fixed_cache[Hp] = torch.fft.rfft2(img)
        return self._fixed_cache[Hp]

    def grid(self, S: int):
        if S not in self._grid_cache:
            yy, xx = torch.meshgrid(torch.arange(S, device=self.device), torch.arange(S, device=self.device), indexing="ij")
            self._grid_cache[S] = (yy.reshape(-1), xx.reshape(-1))
        return self._grid_cache[S]

    def prepare(self, genomes: List[Genome], S: Optional[int] = None) -> GenomeBatch:
        return GenomeBatch(self, genomes, S or self.cfg.world_size)

    # ------------------------------------------------------------------
    def init_state(self, gb: GenomeBatch, envs: List[Environment], seeds: List[int],
                   groups: Optional[List[int]] = None, run_seed: int = 0,
                   genomes: Optional[List[Genome]] = None, fields: Optional[np.ndarray] = None) -> SimState:
        B, C, S, dev = gb.B, gb.C, gb.S, self.device
        if fields is None:
            chm = gb.chm.view(B, C).cpu().numpy()
            fields = np.stack([make_initial_field(envs[i], C, S, seeds[i], chm[i]) for i in range(B)])
        U = torch.as_tensor(fields, device=dev, dtype=torch.float32)
        U = torch.clamp(U, -self.cfg.state_bound, self.cfg.state_bound)
        groups = list(range(B)) if groups is None else list(groups)
        g = torch.Generator(device=dev); g.manual_seed(int(run_seed) & 0x7FFFFFFF)
        Dbuf = U.unsqueeze(0).repeat(self.cfg.delay_max + 1, 1, 1, 1, 1) if self.cfg.delay_max > 0 else None
        R0 = torch.as_tensor([e.resource_scale for e in envs], device=dev, dtype=torch.float32).view(B, 1, 1, 1)
        G = gb.link_w0.expand(B, gb.L, S, S).clone() if gb.L > 0 else None
        return SimState(U=U, Uprev=U.clone(), H=U.clone(), Dbuf=Dbuf, ptr=0, R=R0.expand(B, 1, S, S).clone(), R0=R0,
                        G=G, t=0, bad=torch.zeros(B, dtype=torch.bool, device=dev),
                        group_idx=torch.as_tensor(groups, device=dev, dtype=torch.long), n_groups=max(groups) + 1,
                        gen=g, ext_noise=torch.as_tensor([e.ext_noise for e in envs], device=dev).view(B, 1, 1, 1).float(),
                        shock_period=torch.as_tensor([e.shock_period for e in envs], device=dev, dtype=torch.long),
                        shock_amp=torch.as_tensor([e.shock_amp for e in envs], device=dev).view(B, 1, 1, 1).float(),
                        dt_scale_sum=torch.zeros(B, device=dev), seeds=list(seeds))

    # ------------------------------------------------------------------
    def pad(self, X: torch.Tensor, gb: GenomeBatch, P: int) -> torch.Tensor:
        modes = gb.modes_present
        if len(modes) == 1:
            m = modes[0]
            return F.pad(X, (P, P, P, P), mode=["circular", "reflect", "constant"][m])
        out = torch.zeros(X.shape[0], X.shape[1], X.shape[2] + 2 * P, X.shape[3] + 2 * P, device=X.device, dtype=X.dtype)
        for m in modes:
            sel = (gb.bmode == m).view(-1, 1, 1, 1).to(X.dtype)
            out = out + sel * F.pad(X, (P, P, P, P), mode=["circular", "reflect", "constant"][m])
        return out

    def link_source(self, U: torch.Tensor, gb: GenomeBatch, l: int) -> torch.Tensor:
        B, C, S, _ = U.shape
        yy, xx = self.grid(S)
        ny = yy.unsqueeze(0) + gb.link_oy[:, l:l + 1]
        nx = xx.unsqueeze(0) + gb.link_ox[:, l:l + 1]
        mode = gb.bmode.view(B, 1)
        py, px = ny % S, nx % S
        ry = ny.abs(); ry = torch.where(ry > S - 1, 2 * (S - 1) - ry, ry).clamp(0, S - 1)
        rx = nx.abs(); rx = torch.where(rx > S - 1, 2 * (S - 1) - rx, rx).clamp(0, S - 1)
        valid = ((ny >= 0) & (ny < S) & (nx >= 0) & (nx < S)) | (mode != 2)
        fy = torch.where(mode == 0, py, torch.where(mode == 1, ry, ny.clamp(0, S - 1)))
        fx = torch.where(mode == 0, px, torch.where(mode == 1, rx, nx.clamp(0, S - 1)))
        idx = (fy * S + fx)
        src = torch.gather(U.reshape(B, C, S * S), 2, idx.unsqueeze(1).expand(B, C, S * S))
        src = src * valid.unsqueeze(1).to(U.dtype)
        return src.view(B, C, S, S)

    def _grand(self, st: SimState, shape_tail, kind: str = "normal") -> torch.Tensor:
        """Aleatoriedade compartilhada por grupo (pares contrafactuais recebem o MESMO ruído)."""
        shp = (st.n_groups,) + tuple(shape_tail)
        if kind == "normal":
            r = torch.randn(shp, generator=st.gen, device=self.device)
        else:
            r = torch.rand(shp, generator=st.gen, device=self.device)
        ident = getattr(st, "_groups_identity", None)
        if ident is None:
            ident = st.n_groups == st.U.shape[0] and bool(
                (st.group_idx == torch.arange(st.n_groups, device=self.device)).all())
            st._groups_identity = ident
        if ident:
            return r
        return r.index_select(0, st.group_idx)

    # ------------------------------------------------------------------
    def _physics_eager(self, U, Uprev, H, Ud, N, R, link_sum, W, m0, m1, m2, g_mu, g_sig, g_bias, g_gain, diff, diss, Q,
                       vy, vx, rate, delay_w, hyst_w, link_coup, res_on, res_khalf, K: int):
        Nev = N[:, :K]
        lap, gyy, gxx = N[:, K], N[:, K + 1], N[:, K + 2]
        A = torch.einsum("bckd,bkdhw->bchw", W, Nev)
        x = A + g_bias
        f1 = torch.tanh(x)
        z = (x - g_mu) / g_sig
        f2 = 2.0 * torch.exp(-0.5 * z * z) - 1.0
        f3 = 2.33 * x * torch.exp(-x * x)
        growth = g_gain * (m0 * f1 + m1 * f2 + m2 * f3)
        lim = R / (R + res_khalf)
        gpos = torch.relu(growth)
        growth = res_on * (gpos * lim - torch.relu(-growth)) + (1 - res_on) * growth
        react = U * torch.einsum("bcd,bdhw->bchw", Q, U)
        f = (growth + react + diff * lap - (vy * gyy + vx * gxx) + link_coup * link_sum
             + hyst_w * H + delay_w * Ud - diss * U)
        return rate * f, gpos

    def step(self, gb: GenomeBatch, st: SimState) -> SimState:
        U = st.U
        B, C, S, _ = U.shape
        P, Hp, K = gb.P, gb.Hp, gb.K
        Up = self.pad(U, gb, P)
        Uh = torch.fft.rfft2(Up)
        N = torch.fft.irfft2(Uh.unsqueeze(1) * gb.Khat.unsqueeze(2), s=(Hp, Hp))[..., P:P + S, P:P + S]
        # topologia de longo alcance (campo plástico G)
        link_sum = torch.zeros_like(U)
        new_G = st.G
        if gb.has_links and st.G is not None:
            new_G = st.G.clone()
            for l in range(gb.L):
                if not gb.link_any[l]:
                    continue
                src = self.link_source(U, gb, l)
                on = gb.link_on[:, l].view(B, 1, 1, 1)
                link_sum = link_sum + on * st.G[:, l:l + 1] * src
                if gb.has_plastic:
                    co = (U * src).mean(1, keepdim=True)
                    gl = st.G[:, l:l + 1] + gb.link_eta * (co - gb.link_decay * (st.G[:, l:l + 1] - gb.link_w0[:, l:l + 1]))
                    new_G[:, l:l + 1] = gl.clamp(-2.0, 2.0) * on
        if gb.has_delay and st.Dbuf is not None:
            Dn = st.Dbuf.shape[0]
            idx = ((st.ptr - 1 - gb.delay_k) % Dn).view(1, B, C, 1, 1).expand(1, B, C, S, S)
            Ud = torch.gather(st.Dbuf, 0, idx).squeeze(0)
        else:
            Ud = U
        ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.autocast_dtype is not None else _nullctx()
        args = (U, st.Uprev, st.H, Ud, N, st.R, link_sum, gb.W, gb.m0, gb.m1, gb.m2, gb.g_mu, gb.g_sig, gb.g_bias,
                gb.g_gain, gb.diff, gb.diss, gb.Q, gb.vy, gb.vx, gb.rate, gb.delay_w, gb.hyst_w, gb.link_coup, gb.res_on,
                gb.res_khalf, K)
        # torch.compile só é usado no tamanho fixo do batch de triagem (formas estáticas, sem recompilações
        # a cada probe); os probes, de tamanhos variados, usam o modo eager.
        compiled = self._physics is not self._physics_eager
        fn = self._physics if (compiled and B == self.cfg.batch_size) else self._physics_eager
        try:
            with ctx:
                dU, gpos = fn(*args)
        except Exception as e:
            if fn is self._physics_eager or is_oom_error(e):
                raise
            print(f"[sim] torch.compile falhou em execução ({type(e).__name__}); voltando ao modo eager")
            self._physics = self._physics_eager
            with ctx:
                dU, gpos = self._physics_eager(*args)
        dU = dU.float(); gpos = gpos.float()
        # flags calculadas uma vez por tensor (evita sincronizar GPU->CPU a cada passo)
        if getattr(st, "_ext_src", None) is not st.ext_noise:
            st._ext_src, st._ext_any = st.ext_noise, bool((st.ext_noise > 0).any())
        if getattr(st, "_shock_src", None) is not st.shock_period:
            st._shock_src, st._shock_np = st.shock_period, st.shock_period.cpu().numpy()
        if gb.has_noise or st._ext_any:
            noise_amp = gb.noise * gb.rate.mean(1, keepdim=True).sqrt() + st.ext_noise
            dU = dU + noise_amp * self._grand(st, (C, S, S))
        # choque ambiental periódico (perturbação externa do ambiente)
        sp = st._shock_np
        if (sp > 0).any():
            hit_np = (sp > 0) & ((st.t + 1) % np.maximum(sp, 1) == 0)
            if hit_np.any():
                hit = torch.as_tensor(hit_np, device=self.device, dtype=torch.float32).view(B, 1, 1, 1)
                dU = dU + hit * st.shock_amp * self._grand(st, (C, S, S))
        # dt adaptativo (salvaguarda numérica externa)
        amax = dU.abs().amax(dim=(1, 2, 3), keepdim=True)
        scale = torch.clamp(gb.max_rate / (amax + 1e-12), max=1.0)
        dU = dU * scale
        st.dt_scale_sum = st.dt_scale_sum + scale.view(B)
        if gb.has_async:
            m = (self._grand(st, (1, S, S), "uniform") < gb.async_p).float()
            dU = dU * m
        dU = dU - gb.conserve * dU.mean(dim=(2, 3), keepdim=True)
        Unew = U + dU
        if gb.has_inertia:
            Unew = Unew + gb.inertia * (U - st.Uprev)
        Unew = gb.sat * torch.tanh(Unew / gb.sat)
        Unew = Unew * gb.chm
        bad = ~torch.isfinite(Unew).flatten(1).all(1)
        st.bad = st.bad | bad
        Unew = torch.where(bad.view(B, 1, 1, 1), torch.zeros_like(Unew), Unew)
        Unew = torch.nan_to_num(Unew, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.cfg.state_bound, self.cfg.state_bound)
        # traço lento (histerese)
        if gb.has_hyst:
            st.H = st.H + gb.hyst_rate * (Unew - st.H)
        # recurso abstrato
        if gb.has_res:
            act = gpos.mean(1, keepdim=True)
            Rp = self.pad(st.R, gb, 1)
            Rlap = (Rp[:, :, :-2, 1:-1] + Rp[:, :, 2:, 1:-1] + Rp[:, :, 1:-1, :-2] + Rp[:, :, 1:-1, 2:] - 4 * st.R)
            Rn = st.R + gb.res_rho * (st.R0 - st.R) - gb.res_cons * act * st.R + gb.res_diff * Rlap
            st.R = torch.where(gb.res_on > 0, Rn.clamp(0.0, 4.0), st.R)
        if st.Dbuf is not None:
            st.Dbuf[st.ptr] = Unew
            st.ptr = (st.ptr + 1) % st.Dbuf.shape[0]
        st.Uprev = U
        st.U = Unew
        st.G = new_G
        st.t += 1
        self.steps_done += B
        return st

    def run(self, gb: GenomeBatch, st: SimState, n: int, callback: Optional[Callable] = None) -> SimState:
        for _ in range(int(n)):
            st = self.step(gb, st)
            if callback is not None:
                callback(st)
        return st


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
