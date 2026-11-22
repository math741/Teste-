# %% [markdown]
# ## 8. Escada de descoberta, scorecard, evidência e GENESIS SCIENTIST
# **Fitness ≠ evidência.** Fitness conduz a seleção; evidência mede quanto sabemos (p-valores, replicação, controles).
# A escada é sequencial (um nível só conta se todos os anteriores foram demonstrados) e cada promoção guarda o
# `result_id` que a sustenta (auditabilidade).
#
# O **Scientist** é baseado em regras + valor de informação: prioriza experimentos cujo resultado é mais incerto e mais
# informativo por unidade de custo, nunca repete o mesmo teste de seleção no mesmo candidato (anti p-hacking),
# usa cache por (genoma, probe, versão), evita re-testar comportamentos quase idênticos a falhas já conhecidas
# e dá prioridade a anomalias.

# %%
LEVEL_NAMES = {0: "ACTIVITY", 1: "STRUCTURE", 2: "CAUSALITY", 3: "MEMORY", 4: "CHANNEL", 5: "TRANSFORMATION",
               6: "COMPOSITION", 7: "GENERALIZATION", 8: "ADAPTATION", 9: "EMERGENT COMPUTATIONAL ARCHITECTURE",
               10: "RESERVED"}
SCORECARD_FIELDS = ["activity", "persistence", "mobility", "complexity", "causality", "memory", "channel",
                    "transformation", "composition", "generalization", "recovery", "robustness", "adaptation",
                    "replication", "novelty"]
PROBE_LEVEL = {"causality": 2, "memory": 3, "channel": 4, "transformation": 5, "composition": 6, "generalization": 7,
               "adaptation": 8}
PROBE_COST_DEFAULT = {"causality": 1.0, "memory": 2.0, "channel": 2.0, "transformation": 2.0, "composition": 3.0,
                      "generalization": 3.0, "recovery": 1.5, "robustness": 4.0, "adaptation": 4.0, "replication": 12.0,
                      "ablation": 15.0, "long_screen": 2.0, "advanced": 10.0, "functional_recovery": 2.5,
                      "channel_noise": 2.0, "champion_lab": 60.0}


class DiscoveryLadder:
    def __init__(self, th: Thresholds):
        self.th = th

    def evaluate(self, d: Optional[Dict[str, float]], status: Dict[str, str], replication_ok: bool) -> Tuple[int, List[str]]:
        ev: List[str] = []
        if d is None or d.get("viability") in ("DEAD", "NUMERICALLY_UNSTABLE", "EXPLOSIVE"):
            return -1, ev
        level = -1
        if d.get("rel_activity", 0) >= self.th.frozen_rel_activity or d.get("viability") == "VIABLE":
            level = 0; ev.append("L0: atividade dinâmica na triagem multi-ambiente")
        else:
            return level, ev
        if d.get("persistence", 0) >= self.th.persistence and d.get("structures", 0) >= 1:
            level = 1; ev.append(f"L1: persistência={d['persistence']:.2f}, estruturas≈{d['structures']:.1f}")
        else:
            return level, ev
        for probe, lv in sorted(PROBE_LEVEL.items(), key=lambda x: x[1]):
            if status.get(probe) == STATUS_PASS:
                level = lv; ev.append(f"L{lv}: {probe} PASS")
            else:
                return level, ev
        core = all(status.get(p) == STATUS_PASS for p in ["memory", "channel", "transformation", "composition", "generalization"])
        if core and status.get("robustness") == STATUS_PASS and replication_ok:
            level = 9; ev.append("L9: armazenamento+transmissão+transformação+composição+robustez+generalização+replicação")
        return level, ev


def evidence_score(results: List[TestResult]) -> float:
    """Separado do fitness: soma de evidência estatística ponderada por nível e replicação."""
    s = 0.0
    rep_rate = {r.details.get("target"): r.score for r in results if r.probe == "replication"}
    for r in results:
        if r.status != STATUS_PASS or r.probe not in PROBE_LEVEL:
            continue
        p = r.p_value if (r.p_value is not None and np.isfinite(r.p_value) and r.p_value > 0) else 1e-6
        s += min(10.0, -math.log10(max(p, 1e-10))) * (1 + PROBE_LEVEL[r.probe] / 3) * (0.5 + 0.5 * rep_rate.get(r.probe, 0.0))
    return float(s)


@dataclass
class Experiment:
    candidate_id: str
    probe: str
    kwargs: Dict[str, Any]
    priority: float
    reason: str
    created_gen: int
    exp_id: str = ""

    def __post_init__(self):
        if not self.exp_id:
            self.exp_id = new_id("E")


class GenesisScientist:
    """Decide o próximo experimento. Pluggable: `self.advisor` pode ser trocado por outro pesquisador (ex.: LLM),
    mas nada aqui depende disso."""

    def __init__(self, cfg: GenesisConfig):
        self.cfg, self.th = cfg, cfg.thresholds
        self.queue: List[Experiment] = []
        self.done: Dict[str, str] = {}             # chave (cand|probe|args) -> result_id
        self.cache: Dict[str, str] = {}            # (genome_hash|probe|version|args) -> result_id
        self.stats: Dict[str, Dict[str, float]] = {}
        self.failed_behaviors: Dict[str, List[np.ndarray]] = {}
        self.queued: set = set()                   # chaves já na fila (deduplicação O(1))
        self._fail_cache: Dict[str, Tuple[int, np.ndarray, Any]] = {}
        self.advisor: Optional[Callable] = None

    @staticmethod
    def key(cid: str, probe: str, kwargs: Dict[str, Any]) -> str:
        return f"{cid}|{probe}|{json.dumps(to_jsonable(kwargs), sort_keys=True)}"

    def p_pass(self, probe: str) -> float:
        s = self.stats.get(probe, {"n": 0, "pass": 0})
        return (s["pass"] + 1) / (s["n"] + 2)

    def cost(self, probe: str) -> float:
        s = self.stats.get(probe)
        if s and s.get("n", 0) > 0:
            return max(0.05, s["time"] / s["n"])
        return PROBE_COST_DEFAULT.get(probe, 2.0)

    @staticmethod
    def probe_level(probe: str, kwargs: Dict[str, Any], level: int) -> float:
        """Degrau equivalente: testes de validação valem o nível que validam (replicação/ablação/robustez)."""
        if probe in PROBE_LEVEL:
            return PROBE_LEVEL[probe]
        if probe == "replication":
            return PROBE_LEVEL.get(kwargs.get("target", "memory"), 3)
        if probe in ("robustness", "channel_noise", "functional_recovery"):
            return max(2.0, level)
        if probe == "ablation":
            return 0.7 * PROBE_LEVEL.get(kwargs.get("target", "memory"), 3)
        if probe == "champion_lab":
            return 10.0
        return 1.5

    def value_of_information(self, probe: str, bonus: float, level: int = 0, kwargs: Optional[Dict[str, Any]] = None) -> float:
        """Ganho esperado de conhecimento por custo: incerteza do resultado (p(1-p)), altura do degrau na escada e
        quão promissor o candidato já é (candidatos mais altos na escada recebem os experimentos caros)."""
        p = self.p_pass(probe)
        gain = (1.0 + self.probe_level(probe, kwargs or {}, level)) ** 2
        depth = 1.0 + max(0, level)
        return (gain * (0.25 + p * (1 - p)) * depth + bonus) / (1.0 + self.cost(probe))

    def similar_to_failure(self, probe: str, vec: Optional[np.ndarray]) -> bool:
        lst = self.failed_behaviors.get(probe)
        if vec is None or not lst:
            return False
        ent = self._fail_cache.get(probe)
        if ent is None or ent[0] != len(lst) or ent[2] is not lst[-1]:
            ent = (len(lst), np.asarray(lst[-500:], np.float32), lst[-1])
            self._fail_cache[probe] = ent
        d = np.sqrt(np.mean((ent[1] - np.asarray(vec, np.float32)[None]) ** 2, axis=1))
        return bool((d < 0.015).any())

    def propose(self, cid: str, d: Dict[str, float], status: Dict[str, str], level: int, anomalous: bool, gen: int,
                vec: Optional[np.ndarray] = None, is_top: bool = False) -> List[Experiment]:
        th = self.th
        out: List[Experiment] = []
        bonus = (1.0 if anomalous else 0.0) + 0.25 * max(0, level) + 0.3 * sum(v == STATUS_PASS for v in status.values())

        def want(probe, reason, extra=0.0, **kw):
            k = self.key(cid, probe, kw)
            if k in self.done or k in self.queued or any(self.key(e.candidate_id, e.probe, e.kwargs) == k for e in out):
                return
            if probe in PROBE_LEVEL and not anomalous and self.similar_to_failure(probe, vec):
                return
            out.append(Experiment(cid, probe, kw, self.value_of_information(probe, bonus + extra, level, kw), reason, gen))
        if d.get("viability") not in ("VIABLE", "FROZEN"):
            return out
        if is_top or anomalous:
            want("long_screen", "sobrevivente de triagem (multi-fidelidade)" if is_top else "anomalia", 0.5)
        if "causality" not in status and d.get("causal_spread", 0) > th.causal_spread:
            want("causality", f"causal_spread={d['causal_spread']:.2f}>thr")
        elif "causality" not in status and (level >= 1 and any(v == STATUS_PASS for v in status.values())):
            want("causality", "pré-requisito da escada (outro probe já passou)", 1.0)
        if d.get("persistence", 0) > th.persistence and "recovery" not in status:
            want("recovery", f"persistence={d['persistence']:.2f}>thr")
        if status.get("causality") == STATUS_PASS and "memory" not in status:
            want("memory", "causality PASS -> memory probe", 0.5)
        if d.get("persistence", 0) > th.persistence and "memory" not in status and level >= 1:
            want("memory", "estrutura persistente -> memory probe")
        if status.get("memory") == STATUS_PASS and "channel" not in status:
            want("channel", "memory PASS -> channel probe", 1.0)
        if status.get("channel") == STATUS_PASS:
            if "transformation" not in status:
                want("transformation", "channel PASS -> transformation probe", 1.0)
            if "robustness" not in status:
                want("robustness", "candidato forte -> bateria de robustez")
            if "channel_noise" not in status:
                want("channel_noise", "robustez do canal ao ruído", 0.0, noise_level=self.cfg.probe.channel_noise * 3)
            if status.get("recovery") == STATUS_PASS and "functional_recovery" not in status:
                want("functional_recovery", "recuperação estrutural -> teste funcional pós-dano", 0.5, pre_damage=0.5)
            if "generalization" not in status and status.get("transformation") != STATUS_PASS:
                want("generalization", "channel PASS -> entradas/ambientes não vistos", 0.5, task="channel")
        if status.get("transformation") == STATUS_PASS:
            if "composition" not in status:
                want("composition", "transformation PASS -> composition", 1.0)
            if "generalization" not in status:
                want("generalization", "transformation PASS -> generalization", 1.0, task="xor")
        if status.get("generalization") == STATUS_PASS:
            if "adaptation" not in status:
                want("adaptation", "generalization PASS -> adaptation", 1.0)
            if "replication_channel" not in status:
                want("replication", "descoberta relevante -> replicação independente", 1.0, target="channel")
        if status.get("memory") == STATUS_PASS and "replication_memory" not in status:
            want("replication", "memória -> replicação independente (sementes novas, ambientes de validação)", 0.3,
                 target="memory", n_seeds=self.cfg.probe.replication_seeds)
        passed = [p for p in PROBE_LEVEL if status.get(p) == STATUS_PASS]
        if passed and "ablation" not in status:
            top = max(passed, key=lambda p: PROBE_LEVEL[p])
            if PROBE_LEVEL[top] >= 3:
                want("ablation", f"descoberta {top} -> ablação (o que realmente importa?)", 0.2, target=top)
        if level >= th.champion_level and "champion_lab" not in status:
            want("champion_lab", f"LEVEL {level} -> ChampionLab (bateria final em holdout)", 5.0)
        if self.advisor is not None:
            try:
                out.extend(self.advisor(cid, d, status, level) or [])
            except Exception as e:
                print(f"[scientist] advisor falhou: {e}")
        return out

    EXPLORATION_PROBES = ("causality", "long_screen", "recovery")

    def _eff(self, e: Experiment, gen: int) -> float:
        return e.priority * (1 + 0.15 * max(0, gen - e.created_gen))      # envelhecimento: sem inanição

    def _rebuild_queued(self):
        self.queued = {self.key(e.candidate_id, e.probe, e.kwargs) for e in self.queue}

    def push(self, exps: List[Experiment], gen: int = 0):
        for e in exps:
            k = self.key(e.candidate_id, e.probe, e.kwargs)
            if k in self.queued or k in self.done:
                continue
            self.queue.append(e); self.queued.add(k)
        self.queue.sort(key=lambda e: -self._eff(e, gen))
        if len(self.queue) > 2000:
            self.queue = self.queue[:2000]
            self._rebuild_queued()

    def pop(self, gen: Optional[int] = None) -> Optional[Experiment]:
        """Balanço exploração/exploração: 1 em cada 3 experimentos vai para triagem ampla (exploração),
        os demais para o experimento de maior valor de informação (aprofundamento)."""
        gen = 0 if gen is None else gen
        self._pops = getattr(self, "_pops", 0) + 1
        self.queue.sort(key=lambda e: -self._eff(e, gen))
        explore = self._pops % 3 == 0
        order = ([i for i, e in enumerate(self.queue) if e.probe in self.EXPLORATION_PROBES] if explore else []) + \
            list(range(len(self.queue)))
        for i in order:
            if i >= len(self.queue):
                continue
            e = self.queue[i]
            if self.key(e.candidate_id, e.probe, e.kwargs) in self.done:
                continue
            self.queue.pop(i)
            self.queued.discard(self.key(e.candidate_id, e.probe, e.kwargs))
            return e
        self.queue = [e for e in self.queue if self.key(e.candidate_id, e.probe, e.kwargs) not in self.done]
        self._rebuild_queued()
        return None

    def record(self, e: Experiment, result_id: str, status: str, duration: float, vec: Optional[np.ndarray]):
        self.done[self.key(e.candidate_id, e.probe, e.kwargs)] = result_id
        s = self.stats.setdefault(e.probe, {"n": 0, "pass": 0, "time": 0.0})
        s["n"] += 1; s["time"] += duration; s["pass"] += int(status == STATUS_PASS)
        if status == STATUS_FAIL and vec is not None and e.probe in PROBE_LEVEL:
            self.failed_behaviors.setdefault(e.probe, []).append(vec)
            self.failed_behaviors[e.probe] = self.failed_behaviors[e.probe][-500:]

    def drop_candidates(self, alive: set):
        self.queue = [e for e in self.queue if e.candidate_id in alive]
        self._rebuild_queued()

    def state_dict(self):
        return {"queue": [asdict(e) for e in self.queue], "done": self.done, "cache": self.cache, "stats": self.stats,
                "failed_behaviors": {k: np.asarray(v) for k, v in self.failed_behaviors.items()}}

    def load_state_dict(self, s):
        self.queue = [Experiment(**e) for e in s["queue"]]
        self.done, self.cache, self.stats = dict(s["done"]), dict(s["cache"]), dict(s["stats"])
        self.failed_behaviors = {k: [np.asarray(x) for x in v] for k, v in s["failed_behaviors"].items()}
        self._fail_cache = {}
        self._rebuild_queued()
