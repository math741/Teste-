# %% [markdown]
# ## 02 — Configuração central, perfis e detecção de hardware
# Todos os thresholds e tamanhos ficam aqui (sem números mágicos espalhados). A configuração vai inteira em cada
# checkpoint e em cada descoberta. Perfis: `DEBUG`, `FAST`, `RESEARCH`, `EXTREME`, `A100_80GB` — o autotune ajusta a
# carga real medindo o hardware (nunca assume A100).

# %%
@dataclass
class Thresholds:
    alpha: float = 0.01
    min_effect: float = 0.5            # Cohen's d mínimo para efeitos comportamentais
    min_norm_acc: float = 0.30
    # desenvolvimento / corpo
    dev_min_nodes: int = 4
    dev_canalization: float = 0.55     # IoU médio entre ruídos de desenvolvimento
    body_min_speed: float = 0.004      # unidades de corpo por passo de física (acima do controle passivo)
    # mente
    sensorimotor_gain: float = 0.15    # ganho de sucesso vs sensores ablados
    learning_gain: float = 0.10        # ganho fim-vs-início da vida acima do controle sem plasticidade
    prediction_gain: float = 0.05      # R² extra do estado interno sobre o sensor atual
    # afeto funcional
    regulation_gain: float = 0.15
    valence_acc: float = 0.65
    persistence_ticks: int = 20
    # sociedade / civilização (janelas em ticks de civilização)
    group_min_size: int = 3
    group_persistence: int = 5         # janelas consecutivas
    settlement_min_blocks: int = 6
    settlement_min_agents: int = 4
    settlement_persistence: int = 3
    city_min_population: int = 40
    city_min_blocks: int = 120
    road_min_traffic: float = 0.35
    specialization_p: float = 0.01
    shelter_damage_reduction: float = 0.2
    ethical_negative_fraction: float = 0.8
    ethical_window: int = 400
    # anti-gaming
    metric_exploit_gain: float = 0.15


@dataclass
class ProbeSettings:
    trials: int = 12
    permutations: int = 60
    folds: int = 4
    arena_size: int = 16               # tiles por arena
    arena_ticks: int = 160
    memory_delays: List[int] = field(default_factory=lambda: [1, 10, 40])
    replication_seeds: int = 6
    critical_seeds: int = 20
    damage_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])
    dev_noise_samples: int = 4
    body_test_steps: int = 240


@dataclass
class CivConfig:
    profile: str = "RESEARCH"
    master_seed: int = 20261124
    scientific_replay: bool = False
    human_constraint_experiment: bool = False
    # mundo
    world_w: int = 256
    world_h: int = 256
    chunk: int = 16
    sea_level: float = 0.42
    height_scale: float = 24.0
    n_sound: int = 4
    day_period: int = 240
    season_period: int = 7200
    weather_strength: float = 1.0
    # agentes (F1)
    initial_agents: int = 4000
    max_agents: int = 40000
    n_units: int = 24
    sense_radius: float = 4.0
    max_speed_cap: float = 0.9          # tiles por tick
    min_speed: float = 0.08
    basal_cost: float = 0.004
    move_cost: float = 0.010
    action_cost: float = 0.0012
    signal_cost: float = 0.0006
    transfer_amount: float = 0.02
    build_cost: float = 0.03
    eat_rate: float = 0.12
    eat_efficiency: float = 0.9
    repro_cost_fraction: float = 0.5
    max_age_range: Tuple[int, int] = (1500, 6000)
    carry_capacity: int = 4
    # corpos (F2–F4)
    max_bodies: int = 256
    dev_lattice: int = 5
    n_morph: int = 4
    dev_steps: int = 10
    body_substeps: int = 6
    body_dt: float = 0.02
    body_world_scale: float = 6.0       # ganho corpo→mundo: tiles por unidade de corpo deslocada (coarse-graining
                                        # espaço-temporal declarado; o mesmo ganho em F1 e F2+)
    # construção / matéria
    max_blocks: int = 300000
    max_stack: int = 6
    # multiescala
    region_size: int = 16
    f1_budget_fraction: float = 0.9
    # agenda temporal multiescala (ticks)
    env_every: int = 4
    social_every: int = 10
    culture_every: int = 50
    civ_every: int = 100
    development_every: int = 20
    fidelity_every: int = 50
    aggregate_every: int = 25
    snapshot_every: int = 50
    # evolução QD (ilhas de laboratório)
    n_islands: int = 4
    island_pop: int = 48
    qd_batch: int = 48
    immigrant_every: int = 400
    map_bins: List[int] = field(default_factory=lambda: [6, 6, 5])
    # ciência
    experiment_time_fraction: float = 0.3
    max_queue: int = 400
    max_experiments_per_cycle: int = 3
    # telemetria / servidor
    viewer_port: int = 8765
    telemetry_hz: float = 4.0
    frame_every_seconds: float = 0.25
    max_stream_agents: int = 6000
    snapshot_ring: int = 240
    # sessão
    max_session_hours: float = 24.0
    max_ticks_session: int = 0
    checkpoint_every_minutes: float = 6.0
    stop_on_champion: bool = False
    precision: str = "fp32"
    use_compile: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    probe: ProbeSettings = field(default_factory=ProbeSettings)

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "CivConfig":
        d = copy.deepcopy(d)
        th = Thresholds(**{k: v for k, v in d.pop("thresholds", {}).items() if k in Thresholds.__dataclass_fields__})
        pr = ProbeSettings(**{k: v for k, v in d.pop("probe", {}).items() if k in ProbeSettings.__dataclass_fields__})
        if "max_age_range" in d:
            d["max_age_range"] = tuple(d["max_age_range"])
        cfg = CivConfig(**{k: v for k, v in d.items() if k in CivConfig.__dataclass_fields__})
        cfg.thresholds, cfg.probe = th, pr
        return cfg

    def thresholds_hash(self) -> str:
        return hash_obj(to_jsonable(asdict(self.thresholds)))[:16]


def detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform(), "cpu_count": os.cpu_count(), "timestamp": iso_now(),
                            "device": "cuda" if torch.cuda.is_available() else "cpu", "genesis_version": GENESIS_VERSION}
    if HAVE_PSUTIL:
        try:
            info["ram_total"] = int(psutil.virtual_memory().total)
        except Exception as e:
            info["ram_error"] = repr(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": p.name, "vram_total": int(p.total_memory), "compute_capability": f"{p.major}.{p.minor}"})
        try:
            info["vram_free"] = int(torch.cuda.mem_get_info()[0])
        except Exception:
            info["vram_free"] = int(p.total_memory)
        try:
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            info["bf16_supported"] = False
    else:
        info.update({"gpu_name": "none (CPU)", "vram_total": 0, "vram_free": 0, "bf16_supported": False})
    return info


def choose_profile(requested: str, hw: Dict[str, Any]) -> str:
    if requested != "AUTO":
        return requested
    if hw["device"] != "cuda":
        return "DEBUG"
    name, vram = hw.get("gpu_name", ""), hw.get("vram_total", 0)
    if "A100" in name and vram >= 70e9:
        return "A100_80GB"
    if vram >= 30e9:
        return "EXTREME"
    return "RESEARCH"


def profile_config(profile: str, hw: Dict[str, Any], seed: int) -> CivConfig:
    cfg = CivConfig(profile=profile, master_seed=seed)
    pr = cfg.probe
    if profile == "DEBUG":
        cfg.world_w = cfg.world_h = 64
        cfg.initial_agents, cfg.max_agents, cfg.n_units = 120, 400, 16
        cfg.max_bodies, cfg.dev_lattice, cfg.dev_steps = 12, 4, 8
        cfg.max_blocks, cfg.region_size = 4096, 16
        cfg.n_islands, cfg.island_pop, cfg.qd_batch = 2, 12, 12
        cfg.max_stream_agents, cfg.snapshot_ring = 400, 120
        cfg.checkpoint_every_minutes = 2.0
        cfg.immigrant_every = 200
        pr.trials, pr.permutations, pr.arena_ticks = 8, 30, 96
        pr.memory_delays, pr.replication_seeds, pr.critical_seeds = [1, 8, 24], 3, 6
        pr.body_test_steps, pr.dev_noise_samples = 160, 3
    elif profile == "FAST":
        cfg.world_w = cfg.world_h = 128
        cfg.initial_agents, cfg.max_agents, cfg.n_units = 1200, 6000, 20
        cfg.max_bodies, cfg.dev_lattice, cfg.max_blocks = 64, 5, 60000
        cfg.n_islands, cfg.island_pop = 3, 32
        cfg.checkpoint_every_minutes = 4.0
        pr.trials, pr.permutations = 10, 40
    elif profile == "RESEARCH":
        pass
    elif profile == "EXTREME":
        cfg.world_w = cfg.world_h = 384
        cfg.initial_agents, cfg.max_agents = 12000, 150000
        cfg.max_bodies, cfg.dev_lattice, cfg.max_blocks = 768, 6, 1000000
        cfg.n_islands, cfg.island_pop, cfg.qd_batch = 6, 96, 96
        cfg.max_stream_agents = 12000
        pr.trials, pr.permutations, pr.critical_seeds = 16, 100, 30
    elif profile == "A100_80GB":
        cfg.world_w = cfg.world_h = 512
        cfg.initial_agents, cfg.max_agents, cfg.n_units = 30000, 400000, 32
        cfg.max_bodies, cfg.dev_lattice, cfg.max_blocks = 2048, 6, 3000000
        cfg.n_islands, cfg.island_pop, cfg.qd_batch = 8, 128, 128
        cfg.max_stream_agents = 20000
        pr.trials, pr.permutations, pr.critical_seeds = 20, 120, 30
    else:
        raise ValueError(f"perfil desconhecido: {profile}")
    return cfg
