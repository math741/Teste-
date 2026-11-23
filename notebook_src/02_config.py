# %% [markdown]
# ## 2. Configuração central, perfis de execução e detecção de hardware
# Todos os thresholds ficam centralizados aqui (sem números mágicos espalhados). A configuração é serializável,
# vai em cada checkpoint e em cada descoberta (histórico de configuração / thresholds).

# %%
@dataclass
class Thresholds:
    # sobrevivência dinâmica (fase 1)
    dead_abs: float = 2e-3
    frozen_rel_activity: float = 2e-4
    explosive_saturation: float = 0.6
    noise_tac: float = 0.15
    noise_sac: float = 0.15
    noise_entropy: float = 0.80
    # triagem -> experimentos
    persistence: float = 0.6
    causal_spread: float = 0.10
    anomaly_surprise: float = 4.0
    chaos_lyap: float = 0.03
    # estatística dos probes
    alpha: float = 0.01
    min_norm_acc: float = 0.30
    causal_spread_probe: float = 0.03
    causal_repeatability: float = 0.5
    memory_min_delay: int = 10
    recovery_sim: float = 0.6
    recovery_gain: float = 0.3
    robust_sim: float = 0.75
    robustness: float = 0.6
    generalization_drop: float = 0.30
    adaptation_p: float = 0.05
    replication_pass_rate: float = 0.8
    # champion gate
    champion_level: int = 7
    champion_reproducibility: float = 0.8
    champion_robustness: float = 0.6
    dominance_p: float = 0.05
    dominance_effect: float = 0.5
    # anti-metric-gaming
    metric_exploit_gain: float = 0.10
    metric_exploit_window: int = 15


@dataclass
class ProbeSettings:
    warmup: int = 48
    trials: int = 8
    permutations: int = 100
    folds: int = 4
    memory_delays: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    memory_history_steps: int = 6
    channel_symbols: int = 4
    channel_distance: int = 16
    channel_times: List[int] = field(default_factory=lambda: [0, 2, 4, 8, 16, 32, 64])
    channel_noise: float = 0.02
    stim_radius: int = 3
    obs_radius: int = 4
    stim_amp: float = 0.8
    stim_pulses: int = 2
    transform_times: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    damage_levels: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    recovery_warmup: int = 96
    recovery_steps: int = 96
    robust_steps: int = 128
    robust_trials: int = 4
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.03, 0.1])
    replication_seeds: int = 10
    critical_seeds: int = 30
    causal_steps: int = 64
    causal_trials: int = 8
    adaptation_steps: int = 96


@dataclass
class GenesisConfig:
    profile: str = "RESEARCH"
    # universo
    world_size: int = 64
    channels_max: int = 6
    kernel_slots: int = 3
    radius_max: int = 7
    links_max: int = 4
    delay_max: int = 4
    state_bound: float = 6.0
    # população / triagem
    population_size: int = 128
    batch_size: int = 256
    screen_envs: int = 2
    screen_steps: int = 192
    screen_burn_in: int = 64
    record_every: int = 6
    metric_res: int = 32
    immigrant_rate: float = 0.05
    species_cap_fraction: float = 0.3
    species_threshold: float = 0.35
    novelty_k: int = 10
    novelty_archive_max: int = 6000
    map_bins: List[int] = field(default_factory=lambda: [8, 8, 5])
    map_axes: List[str] = field(default_factory=lambda: ["persistence", "causal_spread", "temporal_complexity"])
    stagnation_window: int = 12
    # numérica
    precision: str = "fp32"
    use_compile: bool = False
    deterministic: bool = False
    # tempo e orçamento
    max_session_hours: float = 11.0
    checkpoint_every_minutes: float = 10.0
    dashboard_every_seconds: float = 30.0
    live_interval: float = 1.0
    max_generations_session: int = 0
    experiment_time_fraction: float = 0.35
    max_experiments_per_generation: int = 6
    champion_lab_max_per_session: int = 2
    # sementes
    master_seed: int = 20261122
    thresholds: Thresholds = field(default_factory=Thresholds)
    probe: ProbeSettings = field(default_factory=ProbeSettings)

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "GenesisConfig":
        d = copy.deepcopy(d)
        th = Thresholds(**{k: v for k, v in d.pop("thresholds", {}).items() if k in Thresholds.__dataclass_fields__})
        pr = ProbeSettings(**{k: v for k, v in d.pop("probe", {}).items() if k in ProbeSettings.__dataclass_fields__})
        cfg = GenesisConfig(**{k: v for k, v in d.items() if k in GenesisConfig.__dataclass_fields__})
        cfg.thresholds, cfg.probe = th, pr
        return cfg

    def thresholds_hash(self) -> str:
        return hash_obj(to_jsonable(asdict(self.thresholds)))[:16]


def detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_version": torch.version.cuda, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "timestamp": iso_now(), "genesis_version": GENESIS_VERSION,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if HAVE_PSUTIL:
        try:
            info["ram_total"] = psutil.virtual_memory().total
        except Exception as e:
            info["ram_total_error"] = repr(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["vram_total"] = int(p.total_memory)
        try:
            free, total = torch.cuda.mem_get_info()
            info["vram_free"] = int(free)
        except Exception:
            info["vram_free"] = int(p.total_memory)
        info["compute_capability"] = f"{p.major}.{p.minor}"
        try:
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            info["bf16_supported"] = False
    else:
        info["gpu_name"] = "none (CPU)"
        info["vram_total"] = 0
        info["vram_free"] = 0
        info["bf16_supported"] = False
    return info


def profile_config(profile: str, hw: Dict[str, Any], master_seed: int, max_hours: float, max_gens: int) -> GenesisConfig:
    """Escolhe defaults seguros por perfil. Tamanho de batch/população é refinado depois pelo autotune."""
    gpu = hw["device"] == "cuda"
    if profile == "AUTO":
        if not gpu:
            profile = "DEBUG"
        else:
            profile = "RESEARCH"
    cfg = GenesisConfig(profile=profile, master_seed=master_seed, max_session_hours=max_hours,
                        max_generations_session=max_gens)
    pr = cfg.probe
    if profile == "DEBUG":
        cfg.world_size, cfg.channels_max, cfg.kernel_slots, cfg.radius_max = 32, 4, 2, 5
        cfg.links_max, cfg.delay_max = 2, 2
        cfg.population_size, cfg.batch_size = 24, 48
        cfg.screen_envs, cfg.screen_steps, cfg.screen_burn_in, cfg.record_every, cfg.metric_res = 2, 96, 32, 4, 16
        cfg.map_bins = [6, 6, 4]
        cfg.checkpoint_every_minutes, cfg.dashboard_every_seconds = 2.0, 10.0
        cfg.max_experiments_per_generation = 3
        cfg.stagnation_window = 6
        pr.trials, pr.permutations, pr.warmup = 8, 40, 24
        pr.memory_delays, pr.channel_distance = [1, 10, 40], 8
        pr.channel_times, pr.transform_times = [0, 2, 4, 8, 16, 32], [4, 8, 16, 32]
        pr.recovery_warmup, pr.recovery_steps, pr.robust_steps, pr.robust_trials = 48, 48, 64, 3
        pr.replication_seeds, pr.critical_seeds, pr.causal_steps, pr.causal_trials = 4, 6, 32, 6
        pr.adaptation_steps, pr.stim_radius, pr.obs_radius = 48, 2, 3
        pr.damage_levels = [0.10, 0.50, 0.90]
        pr.noise_levels = [0.0, 0.03, 0.1]
    elif profile == "FAST":
        cfg.world_size, cfg.channels_max = 48, 5
        cfg.population_size, cfg.batch_size = 96, 256
        cfg.screen_steps, cfg.screen_burn_in, cfg.record_every, cfg.metric_res = 160, 48, 6, 24
        cfg.checkpoint_every_minutes = 5.0
        cfg.dashboard_every_seconds = 15.0
        pr.trials, pr.permutations = 8, 60
        pr.memory_delays, pr.channel_distance = [1, 10, 100], 12
        pr.replication_seeds, pr.critical_seeds = 6, 12
    elif profile == "RESEARCH":
        cfg.world_size = 64
        cfg.population_size, cfg.batch_size = 256, 512
        cfg.screen_steps, cfg.screen_burn_in = 224, 64
        cfg.checkpoint_every_minutes = 5.0
        pr.memory_delays = [1, 10, 100, 1000]
    elif profile == "EXTREME":
        cfg.world_size = 96 if hw.get("vram_total", 0) >= 20e9 else 64
        cfg.population_size, cfg.batch_size = 512, 1024
        cfg.screen_steps, cfg.screen_burn_in = 288, 96
        cfg.checkpoint_every_minutes = 5.0
        pr.memory_delays = [1, 10, 100, 1000, 10000]
        pr.trials, pr.permutations = 12, 200
        pr.critical_seeds = 30
    else:
        raise ValueError(f"perfil desconhecido: {profile}")
    return cfg


VRAM_TARGET = {"DEBUG": 0.30, "FAST": 0.50, "RESEARCH": 0.70, "EXTREME": 0.87}
