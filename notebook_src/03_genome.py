# %% [markdown]
# ## 3. Genoma de universo
# O genoma é um conjunto estruturado de **leis** (não de "programa"): kernels espaciais (forma, raio, anisotropia),
# tensor de interação entre canais, família de não linearidades, difusão, dissipação, reação, transporte, inércia (ondas),
# constantes temporais, atraso, histerese, grau de conservação, saturação, topologia de longo alcance e sua plasticidade,
# dinâmica de recurso abstrato, condição de borda, assincronia e os **próprios parâmetros de mutação** (auto-adaptativos).
#
# Nenhum canal tem significado. Os nomes abaixo descrevem **o papel matemático do termo na lei**, não a semântica do estado.

# %%
KERNEL_SHAPES = ["ring", "gaussian", "moore", "von_neumann", "anisotropic_ring"]
BOUNDARIES = ["periodic", "reflective", "absorbing"]
MUTATION_OPERATORS = ["gauss_small", "gauss_strong", "structural", "temporal", "topology", "crossover"]


@dataclass
class ParamSpec:
    name: str
    shape: Tuple[int, ...]
    low: float
    high: float
    kind: str          # float | int | bool | cat | logfloat
    group: str         # local_laws, interaction, nonlinear, diffusion, ...
    channel_axes: Tuple[int, ...] = ()


class ParamSpace:
    """Define o espaço de genomas, amostragem, limites, mutação e crossover."""

    def __init__(self, cfg: GenesisConfig):
        C, K, L, D = cfg.channels_max, cfg.kernel_slots, cfg.links_max, cfg.delay_max
        R = cfg.radius_max
        self.C, self.K, self.L, self.D, self.R = C, K, L, D, R
        S = [
            ParamSpec("ch_active", (C,), 0, 1, "bool", "channels", (0,)),
            ParamSpec("k_active", (K,), 0, 1, "bool", "topology"),
            ParamSpec("k_shape", (K,), 0, len(KERNEL_SHAPES) - 1, "cat", "topology"),
            ParamSpec("k_radius", (K,), 1.0, float(R), "float", "topology"),
            ParamSpec("k_mu", (K,), 0.1, 0.9, "float", "topology"),
            ParamSpec("k_sigma", (K,), 0.05, 0.5, "float", "topology"),
            ParamSpec("k_aniso", (K,), 0.3, 1.0, "float", "topology"),
            ParamSpec("k_angle", (K,), 0.0, math.pi, "float", "topology"),
            ParamSpec("W", (C, K, C), -2.0, 2.0, "float", "interaction", (0, 2)),
            ParamSpec("g_mix", (C, 3), -1.5, 1.5, "float", "nonlinear", (0,)),
            ParamSpec("g_mu", (C,), -1.0, 1.5, "float", "nonlinear", (0,)),
            ParamSpec("g_sigma", (C,), 0.05, 1.0, "float", "nonlinear", (0,)),
            ParamSpec("g_bias", (C,), -1.0, 1.0, "float", "nonlinear", (0,)),
            ParamSpec("g_gain", (C,), 0.0, 2.0, "float", "nonlinear", (0,)),
            ParamSpec("diff", (C,), 0.0, 1.0, "float", "diffusion", (0,)),
            ParamSpec("diss", (C,), 0.0, 1.0, "float", "dissipation", (0,)),
            ParamSpec("Q", (C, C), -1.0, 1.0, "float", "reaction", (0, 1)),
            ParamSpec("vel", (C, 2), -0.5, 0.5, "float", "propagation", (0,)),
            ParamSpec("inertia", (C,), 0.0, 0.95, "float", "propagation", (0,)),
            ParamSpec("tau", (C,), 0.5, 20.0, "logfloat", "temporal", (0,)),
            ParamSpec("delay_idx", (C,), 0, D, "int", "temporal", (0,)),
            ParamSpec("delay_w", (C,), -1.0, 1.0, "float", "temporal", (0,)),
            ParamSpec("hyst_rate", (C,), 0.001, 0.5, "logfloat", "temporal", (0,)),
            ParamSpec("hyst_w", (C,), -1.0, 1.0, "float", "temporal", (0,)),
            ParamSpec("conserve", (C,), 0.0, 1.0, "float", "conservation", (0,)),
            ParamSpec("sat", (C,), 0.5, 4.0, "float", "saturation", (0,)),
            ParamSpec("link_coup", (C,), -1.0, 1.0, "float", "topology", (0,)),
            ParamSpec("link_active", (L,), 0, 1, "bool", "topology"),
            ParamSpec("link_offset", (L, 2), -0.5, 0.5, "float", "topology"),
            ParamSpec("link_weight", (L,), -1.0, 1.0, "float", "topology"),
            ParamSpec("link_eta", (), 0.0, 0.2, "float", "topology"),
            ParamSpec("link_decay", (), 0.0, 0.2, "float", "topology"),
            ParamSpec("res_enabled", (), 0, 1, "bool", "resource_dynamics"),
            ParamSpec("res_rho", (), 0.0, 0.2, "float", "resource_dynamics"),
            ParamSpec("res_cons", (), 0.0, 1.0, "float", "resource_dynamics"),
            ParamSpec("res_khalf", (), 0.05, 2.0, "logfloat", "resource_dynamics"),
            ParamSpec("res_diff", (), 0.0, 0.2, "float", "resource_dynamics"),
            ParamSpec("max_rate", (), 0.05, 1.0, "float", "local_laws"),
            ParamSpec("noise", (), 0.0, 0.05, "float", "local_laws"),
            ParamSpec("async_p", (), 0.5, 1.0, "float", "temporal"),
            ParamSpec("time_scale", (), 0.05, 0.5, "float", "temporal"),
            ParamSpec("boundary", (), 0, len(BOUNDARIES) - 1, "cat", "topology"),
            ParamSpec("mut_sigma", (), 0.005, 0.5, "logfloat", "mutation_parameters"),
            ParamSpec("mut_strong", (), 0.0, 0.3, "float", "mutation_parameters"),
            ParamSpec("mut_struct", (), 0.0, 0.5, "float", "mutation_parameters"),
            ParamSpec("mut_topo", (), 0.0, 0.5, "float", "mutation_parameters"),
        ]
        self.specs: Dict[str, ParamSpec] = {s.name: s for s in S}
        self.mutation_param_names = ["mut_sigma", "mut_strong", "mut_struct", "mut_topo"]
        self.groups = sorted(set(s.group for s in S))

    # ------------------------------------------------------------------
    def clamp(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for n, s in self.specs.items():
            v = np.asarray(params[n])
            if s.kind == "bool":
                v = v.astype(bool)
            elif s.kind in ("int", "cat"):
                v = np.clip(np.rint(np.nan_to_num(v.astype(np.float64))), s.low, s.high).astype(np.int64)
            else:
                v = np.clip(np.nan_to_num(v.astype(np.float64), nan=(s.low + s.high) / 2), s.low, s.high).astype(np.float32)
            params[n] = np.array(v).reshape(s.shape)
        if not params["ch_active"].any():
            params["ch_active"][0] = True
        if not params["k_active"].any():
            params["k_active"][0] = True
        return params

    def random_params(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        p: Dict[str, np.ndarray] = {}
        for n, s in self.specs.items():
            if s.kind == "bool":
                p[n] = rng.random(s.shape) < 0.6
            elif s.kind in ("int", "cat"):
                p[n] = rng.integers(int(s.low), int(s.high) + 1, size=s.shape)
            elif s.kind == "logfloat":
                p[n] = np.exp(rng.uniform(math.log(s.low), math.log(s.high), size=s.shape))
            else:
                p[n] = rng.uniform(s.low, s.high, size=s.shape)
        C = self.C
        n_ch = int(rng.integers(2, C + 1)) if C >= 2 else 1
        act = np.zeros(C, bool); act[rng.choice(C, n_ch, replace=False)] = True
        p["ch_active"] = act
        # priors mais suaves para leis aleatórias (a maioria ainda será lixo — é para isso que existe seleção)
        p["W"] = rng.normal(0, 0.6, size=self.specs["W"].shape) * (rng.random(self.specs["W"].shape) < 0.6)
        p["Q"] = rng.normal(0, 0.3, size=self.specs["Q"].shape) * (rng.random(self.specs["Q"].shape) < 0.3)
        p["diss"] = rng.uniform(0.0, 0.4, size=(C,))
        p["diff"] = rng.uniform(0.0, 0.4, size=(C,))
        p["vel"] = rng.normal(0, 0.1, size=(C, 2))
        p["inertia"] = rng.uniform(0, 0.5, size=(C,)) * (rng.random(C) < 0.4)
        p["delay_w"] = rng.normal(0, 0.3, size=(C,)) * (rng.random(C) < 0.3)
        p["hyst_w"] = rng.normal(0, 0.3, size=(C,)) * (rng.random(C) < 0.3)
        p["conserve"] = rng.uniform(0, 1, size=(C,)) * (rng.random(C) < 0.4)
        p["link_active"] = rng.random(self.L) < 0.3
        p["link_eta"] = np.float32(rng.uniform(0, 0.05) if rng.random() < 0.3 else 0.0)
        p["noise"] = np.float32(rng.uniform(0, 0.01))
        p["async_p"] = np.float32(1.0 if rng.random() < 0.6 else rng.uniform(0.5, 1.0))
        p["mut_sigma"] = np.float32(0.05)
        p["mut_strong"] = np.float32(0.05)
        p["mut_struct"] = np.float32(0.1)
        p["mut_topo"] = np.float32(0.1)
        p["boundary"] = np.int64(rng.choice([0, 0, 1, 2]))
        return self.clamp(p)

    def zero_params(self) -> Dict[str, np.ndarray]:
        """Lei nula (baseline/null model): nada interage, nada muda exceto saturação suave."""
        p = {}
        for n, s in self.specs.items():
            if s.kind == "bool":
                p[n] = np.zeros(s.shape, bool)
            elif s.kind in ("int", "cat"):
                p[n] = np.zeros(s.shape, np.int64)
            else:
                p[n] = np.full(s.shape, s.low, np.float32) if s.low > 0 else np.zeros(s.shape, np.float32)
        p["ch_active"][:] = True
        p["k_active"][0] = True
        p["W"][:] = 0; p["g_gain"][:] = 0; p["diss"][:] = 0; p["diff"][:] = 0
        p["sat"][:] = 4.0; p["tau"][:] = 1.0; p["time_scale"] = np.float32(0.2)
        p["max_rate"] = np.float32(1.0); p["async_p"] = np.float32(1.0)
        p["mut_sigma"] = np.float32(0.05)
        return self.clamp(p)

    # ------------------------------------------------------------------
    def vector(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Vetor normalizado do genótipo (para especiação/distância genética)."""
        out = []
        for n, s in self.specs.items():
            if s.group == "mutation_parameters":
                continue
            v = np.asarray(params[n], dtype=np.float64).ravel()
            if s.kind == "logfloat":
                v = (np.log(v) - math.log(s.low)) / (math.log(s.high) - math.log(s.low))
            elif s.high > s.low:
                v = (v - s.low) / (s.high - s.low)
            out.append(v)
        return np.concatenate(out).astype(np.float32)

    # ------------------------------------------------------------------
    def _perturb(self, p, name, rng, sigma, frac=0.5):
        s = self.specs[name]
        v = np.asarray(p[name], dtype=np.float64)
        mask = rng.random(s.shape) < frac if s.shape else np.array(True)
        if s.kind == "logfloat":
            v = np.exp(np.log(v) + mask * rng.normal(0, sigma * 2.0, size=s.shape))
        elif s.kind in ("int", "cat"):
            v = v + mask * rng.integers(-1, 2, size=s.shape)
        elif s.kind == "bool":
            flip = rng.random(s.shape) < sigma * frac if s.shape else (rng.random() < sigma * frac)
            v = np.logical_xor(v.astype(bool), flip)
        else:
            v = v + mask * rng.normal(0, sigma * (s.high - s.low), size=s.shape)
        p[name] = v

    def _resample(self, p, name, rng, frac=0.3):
        s = self.specs[name]
        fresh = self.random_params(rng)[name]
        mask = rng.random(s.shape) < frac if s.shape else np.array(True)
        p[name] = np.where(mask, fresh, p[name])

    def duplicate_channel(self, p, rng) -> str:
        act = np.where(p["ch_active"])[0]; ina = np.where(~p["ch_active"])[0]
        if act.size == 0 or ina.size == 0:
            return "dup_channel(skip)"
        src, dst = int(rng.choice(act)), int(rng.choice(ina))
        for n, s in self.specs.items():
            for ax in s.channel_axes:
                a = np.array(p[n])
                idx_src = [slice(None)] * a.ndim; idx_src[ax] = src
                idx_dst = [slice(None)] * a.ndim; idx_dst[ax] = dst
                a[tuple(idx_dst)] = a[tuple(idx_src)]
                p[n] = a
        p["ch_active"][dst] = True
        # pequena quebra de simetria
        self._perturb(p, "W", rng, 0.02, frac=0.2)
        return f"dup_channel({src}->{dst})"

    def remove_channel(self, p, rng) -> str:
        act = np.where(p["ch_active"])[0]
        if act.size <= 1:
            return "remove_channel(skip)"
        c = int(rng.choice(act))
        p["ch_active"][c] = False
        return f"remove_channel({c})"

    def mutate(self, params: Dict[str, np.ndarray], rng: np.random.Generator, op: str,
               sigma_mult: float = 1.0, other: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, np.ndarray], List[str]]:
        p = {k: np.array(v, copy=True) for k, v in params.items()}
        rec: List[str] = []
        # auto-adaptação dos parâmetros de mutação (log-normal) — meta-evolução restrita ao genoma
        tau_m = 0.2
        p["mut_sigma"] = np.float32(float(p["mut_sigma"]) * math.exp(tau_m * rng.normal()))
        for n in ["mut_strong", "mut_struct", "mut_topo"]:
            p[n] = np.float32(float(p[n]) * math.exp(tau_m * rng.normal()) + 0.002 * rng.normal())
        p = self.clamp(p)
        sigma = float(p["mut_sigma"]) * sigma_mult
        float_names = [n for n, s in self.specs.items() if s.kind in ("float", "logfloat") and s.group != "mutation_parameters"]
        if op == "gauss_small":
            chosen = [n for n in float_names if rng.random() < 0.35] or [str(rng.choice(float_names))]
            for n in chosen:
                self._perturb(p, n, rng, sigma)
            rec.append(f"gauss_small(sigma={sigma:.3f},n={len(chosen)})")
        elif op == "gauss_strong":
            chosen = list(rng.choice(float_names, size=int(rng.integers(1, 4)), replace=False))
            for n in chosen:
                self._resample(p, n, rng, frac=0.5)
            rec.append(f"gauss_strong({','.join(chosen)})")
        elif op == "structural":
            r = rng.random()
            if r < 0.25:
                rec.append(self.duplicate_channel(p, rng))
            elif r < 0.40:
                rec.append(self.remove_channel(p, rng))
            elif r < 0.60:
                k = int(rng.integers(0, self.K))
                p["k_shape"][k] = int(rng.integers(0, len(KERNEL_SHAPES)))
                p["k_active"][k] = True
                rec.append(f"kernel_shape(k={k}->{KERNEL_SHAPES[p['k_shape'][k]]})")
            elif r < 0.75:
                k = int(rng.integers(0, self.K))
                p["k_active"][k] = ~p["k_active"][k]
                rec.append(f"kernel_toggle(k={k})")
            else:
                mask = rng.random(self.specs["W"].shape) < 0.15
                p["W"] = np.where(mask, rng.normal(0, 0.8, size=mask.shape) * (rng.random(mask.shape) < 0.7), p["W"])
                rec.append("interaction_rewire")
        elif op == "temporal":
            for n in ["tau", "inertia", "delay_w", "hyst_rate", "hyst_w", "time_scale", "async_p"]:
                if rng.random() < 0.5:
                    self._perturb(p, n, rng, sigma)
            if rng.random() < 0.3:
                self._perturb(p, "delay_idx", rng, 1.0)
            rec.append("temporal")
        elif op == "topology":
            r = rng.random()
            if r < 0.3:
                l = int(rng.integers(0, self.L)); p["link_active"][l] = ~p["link_active"][l]
                rec.append(f"link_toggle(l={l})")
            elif r < 0.6:
                self._perturb(p, "link_offset", rng, sigma * 2); self._perturb(p, "link_weight", rng, sigma)
                rec.append("link_shift")
            elif r < 0.75:
                self._perturb(p, "link_eta", rng, sigma); self._perturb(p, "link_decay", rng, sigma)
                rec.append("link_plasticity")
            elif r < 0.92:
                self._perturb(p, "k_radius", rng, sigma * 2); self._perturb(p, "k_aniso", rng, sigma)
                self._perturb(p, "k_angle", rng, sigma)
                rec.append("kernel_geometry")
            else:
                p["boundary"] = np.int64(rng.integers(0, len(BOUNDARIES)))
                rec.append(f"boundary->{BOUNDARIES[int(p['boundary'])]}")
        elif op == "crossover" and other is not None:
            groups = sorted(set(s.group for s in self.specs.values()))
            taken = []
            for g in groups:
                if g == "mutation_parameters":
                    continue
                if rng.random() < 0.5:
                    for n, s in self.specs.items():
                        if s.group == g:
                            p[n] = np.array(other[n], copy=True)
                    taken.append(g)
            self._perturb(p, str(rng.choice(float_names)), rng, sigma * 0.5)
            rec.append(f"crossover(groups={'+'.join(taken) or 'none'})")
        else:
            self._perturb(p, str(rng.choice(float_names)), rng, sigma)
            rec.append("gauss_single")
        # mutação estrutural / topológica ocasional segundo as taxas herdadas
        if op != "structural" and rng.random() < float(p["mut_struct"]) * 0.3:
            rec.append(self.duplicate_channel(p, rng) if rng.random() < 0.5 else self.remove_channel(p, rng))
        if op != "topology" and rng.random() < float(p["mut_topo"]) * 0.3:
            l = int(rng.integers(0, self.L)); p["link_active"][l] = ~p["link_active"][l]
            rec.append(f"link_toggle(l={l})")
        return self.clamp(p), rec


@dataclass
class Genome:
    params: Dict[str, np.ndarray]
    genome_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    mutation_record: List[str] = field(default_factory=list)
    birth_timestamp: float = field(default_factory=now_ts)
    species_id: int = -1
    origin: str = "random"
    operator: str = "init"

    def __post_init__(self):
        if not self.genome_id:
            self.genome_id = new_id("G")

    def hash(self) -> str:
        return hash_obj(self.params)[:20]

    def n_channels(self) -> int:
        return int(np.asarray(self.params["ch_active"]).sum())

    def to_dict(self) -> dict:
        return {"params": {k: np.array(v) for k, v in self.params.items()}, "genome_id": self.genome_id,
                "parent_ids": list(self.parent_ids), "generation": self.generation,
                "mutation_record": list(self.mutation_record), "birth_timestamp": self.birth_timestamp,
                "species_id": self.species_id, "origin": self.origin, "operator": self.operator}

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        return Genome(params={k: np.array(v) for k, v in d["params"].items()}, genome_id=d["genome_id"],
                      parent_ids=list(d.get("parent_ids", [])), generation=int(d.get("generation", 0)),
                      mutation_record=list(d.get("mutation_record", [])),
                      birth_timestamp=float(d.get("birth_timestamp", now_ts())),
                      species_id=int(d.get("species_id", -1)), origin=d.get("origin", "imported"),
                      operator=d.get("operator", "init"))

    def summary(self, space: "ParamSpace") -> str:
        p = self.params
        act = np.where(p["ch_active"])[0].tolist()
        ks = [f"{KERNEL_SHAPES[int(p['k_shape'][k])]}(r={float(p['k_radius'][k]):.1f})"
              for k in range(space.K) if p["k_active"][k]]
        links = int(np.asarray(p["link_active"]).sum())
        return (f"{self.genome_id} gen={self.generation} ch={act} kernels={ks} links={links} "
                f"boundary={BOUNDARIES[int(p['boundary'])]} dt={float(p['time_scale']):.3f} "
                f"mut_sigma={float(p['mut_sigma']):.3f} origin={self.origin}")


def control_genome(space: ParamSpace, kind: str, channel_distance: int = 8, world_size: int = 32) -> Genome:
    """Leis artificiais conhecidas, usadas SOMENTE como controles positivos/negativos dos instrumentos
    e como baselines (lei simples, lei nula, ruído). Não entram na evolução."""
    p = space.zero_params()
    if kind == "identity":          # preserva o estado (controle positivo de memória / negativo de causalidade)
        p["time_scale"] = np.float32(0.2)     # lei nula: só a saturação suave age
    elif kind == "noise_replace":   # substitui o estado por ruído a cada passo (controle negativo de memória)
        p["time_scale"] = np.float32(0.5); p["tau"][:] = 0.5; p["diss"][:] = 1.0
        p["noise"] = np.float32(0.05); p["max_rate"] = np.float32(1.0)
    elif kind == "diffusion":       # difusão pura (controle positivo de causalidade/recuperação)
        p["time_scale"] = np.float32(0.5); p["tau"][:] = 0.5; p["diff"][:] = 0.2
    elif kind == "link_copy":       # conexão distante que copia a região A em B (controle positivo de canal)
        p["time_scale"] = np.float32(0.5); p["tau"][:] = 0.5; p["diss"][:] = 1.0
        p["link_active"][:] = False; p["link_active"][0] = True
        p["link_offset"][0] = [0.0, -channel_distance / float(world_size)]
        p["link_weight"][0] = 1.0; p["link_coup"][:] = 1.0
    elif kind == "reaction_diffusion":   # baseline de lei simples (tipo ativador-inibidor)
        p["time_scale"] = np.float32(0.2)
        p["diff"][:] = 0.05; p["diff"][1 % space.C] = 0.4
        p["W"][0, 0, 0] = 1.2; p["W"][0, 0, 1 % space.C] = -1.5
        p["W"][1 % space.C, 0, 0] = 1.0
        p["g_gain"][:] = 1.0; p["g_mix"][:, 0] = 1.0; p["diss"][:] = 0.3
        p["k_shape"][0] = 0; p["k_radius"][0] = min(4.0, float(space.R)); p["k_mu"][0] = 0.5; p["k_sigma"][0] = 0.15
    else:
        raise ValueError(kind)
    return Genome(params=space.clamp(p), origin=f"control:{kind}")
