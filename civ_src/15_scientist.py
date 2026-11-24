# %% [markdown]
# ## 15 — SCIENTIST ENGINE · ARCHIVE ENGINE · EXPERIMENT DSL
# **Scientist**: decide quais candidatos merecem testes caros. Fila **limitada** com prioridade adaptativa (nível,
# novidade, incerteza, valor de informação, importância histórica, idade), *gating* da escada (pré-requisitos) com
# triagem exploratória do próximo nível, orçamento adaptativo (fila cheia → mais validação; vazia → mais
# exploração) e respiração científica `EXPLORE → FIND → INVESTIGATE → VALIDATE → EXPLORE`.
# Princípio de descoberta: passou na triagem → **tentar quebrar** em sementes de holdout independentes (replicação
# multi-seed) → só então promover a descoberta (pacote completo + Pantheon revogável). Fitness ≠ evidência.
# **ArchiveEngine**: zoológico de candidatos, arquivo de falhas (resultados negativos são salvos), arquivo de
# estranheza (anomalias nunca são todas descartadas), Pantheon multi-eixo, pacotes de descoberta.
# **DSL**: experimentos só compõem operações seguras (`perturb freeze ablate duplicate fork change_environment observe
# measure`) com validação de esquema; nunca Python arbitrário; opera só no universo simulado.

# %%
LEVEL_WEIGHT = {c: i + 1 for i, c in enumerate(H_CODES)}
LEVEL_WEIGHT.update({c: 2 + i for i, c in enumerate(A_CODES)})
PANTHEON_AXES = {"development": ["H0", "H1"], "learning": ["H3"], "memory": ["H4"], "generalization": ["H5", "H6"],
                 "problem_solving": ["H7"], "affect": A_CODES, "communication": ["H9"], "culture": ["H8", "H10", "H11"],
                 "adaptation": ["H12"]}


@dataclass
class Candidate:
    cid: str
    genome: np.ndarray
    origin: str
    first_seen: str
    tick: int = 0
    fitness: float = 0.0
    novelty: float = 0.0
    importance: float = 0.0
    passed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed: Dict[str, int] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)
    screened: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    island: int = -1

    def evidence_score(self) -> float:
        return float(sum(LEVEL_WEIGHT.get(c, 1) * min(1.0, abs(v.get("effect", 0.0)) / 2 + 0.25) for c, v in self.passed.items()))

    def h_level(self) -> int:
        lvl = -1
        for i, c in enumerate(H_CODES):
            if c in self.passed:
                lvl = i
            else:
                break
        return lvl

    def a_level(self) -> int:
        lvl = -1
        for i, c in enumerate(A_CODES):
            if c in self.passed:
                lvl = i
            else:
                break
        return lvl


class ArchiveEngine:
    def __init__(self, paths: Optional[CivPaths], cfg: CivConfig):
        self.paths, self.cfg = paths, cfg
        self.zoo: "collections.OrderedDict[str, Candidate]" = collections.OrderedDict()
        self.failures: "collections.deque" = collections.deque(maxlen=5000)
        self.weird: "collections.deque" = collections.deque(maxlen=500)
        self.pantheon: Dict[str, Dict[str, Any]] = {}
        self.discoveries: List[Dict[str, Any]] = []
        self.convergence: List[Dict[str, Any]] = []
        self.overall_champion: Optional[Dict[str, Any]] = None

    def add_candidate(self, c: Candidate, cap: int = 3000):
        self.zoo[c.cid] = c
        self.zoo.move_to_end(c.cid)
        while len(self.zoo) > cap:
            # remove o menos importante, mas nunca quem tem evidência
            victim = next((k for k, v in self.zoo.items() if not v.passed), None)
            if victim is None:
                break
            del self.zoo[victim]

    def failure(self, cid: str, res: ProbeResult):
        self.failures.append({"cid": cid, "level": res.level, "p": res.p, "effect": res.effect, "ts": res.ts,
                              "metrics": {k: v for k, v in res.metrics.items() if not isinstance(v, (list, dict))}})

    def anomaly(self, kind: str, data: Dict[str, Any]):
        self.weird.append({"kind": kind, "ts": iso_now(), **to_jsonable(data)})

    # ------------------------------------------------------------------ pacote de descoberta
    def save_discovery(self, level: str, cand: Candidate, results: List[ProbeResult], cfg: CivConfig, tick: int,
                       kind: str = "individual", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        did = f"D{len(self.discoveries) + 1:05d}-{level}-{uuid.uuid4().hex[:6]}"
        meta = {"id": did, "level": level, "level_name": LEVEL_NAME.get(level, level), "kind": kind, "candidate": cand.cid,
                "origin": cand.origin, "tick": tick, "ts": iso_now(), "claim": results[0].claim if results else "TESTED",
                "replications": len(results), "passed_replications": sum(r.passed for r in results),
                "p_values": [r.p for r in results], "effects": [r.effect for r in results],
                "holm_adjusted": holm_correction([r.p for r in results]), "versions": ENGINE_VERSIONS,
                "versions_hash": ENGINE_VERSIONS_HASH, "thresholds_hash": cfg.thresholds_hash(),
                "validation_protocol": VALIDATION_ENGINE_VERSION, "statement": "comportamento observado + mecanismo funcional "
                "demonstrado por probe com controles; não implica experiência subjetiva, vida real ou equivalência humana",
                **(extra or {})}
        meta["evidence"] = [r.public() for r in results]
        if self.paths is not None:
            d = os.path.join(self.paths.discoveries, did)
            os.makedirs(os.path.join(d, "plots"), exist_ok=True)
            np.save(os.path.join(d, "genome.npy"), cand.genome.astype(np.float32))
            files = {"metadata.json": meta, "seeds.json": {"seeds": [r.seeds for r in results]},
                     "config.json": cfg.to_dict(), "evidence.json": meta["evidence"],
                     "statistics.json": {"p": meta["p_values"], "effects": meta["effects"], "holm": meta["holm_adjusted"],
                                         "ci": [r.ci for r in results]},
                     "ablation.json": [r.controls for r in results], "version.json": ENGINE_VERSIONS}
            for name, obj in files.items():
                with open(os.path.join(d, name), "w") as f:
                    json.dump(to_jsonable(obj), f, indent=1)
            rp = next((r.replay for r in results if r.replay), None)
            if rp:
                np.savez_compressed(os.path.join(d, "replay.npz"), pos=rp["pos"], arena_of=rp["arena_of"],
                                    size=np.asarray(rp["size"]), ticks=np.asarray(rp["ticks"]))
                with open(os.path.join(d, "replay_layout.json"), "w") as f:
                    json.dump(to_jsonable(rp["layout"]), f)
            try:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].bar(range(len(results)), [r.effect for r in results], color="#5b8def"); ax[0].set_title("effect size")
                ax[1].bar(range(len(results)), [-math.log10(max(1e-12, r.p)) if np.isfinite(r.p) else 0 for r in results],
                          color="#46c3a0")
                ax[1].axhline(-math.log10(cfg.thresholds.alpha), color="#e0614c", ls="--"); ax[1].set_title("-log10 p")
                fig.suptitle(f"{did}"); fig.tight_layout(); fig.savefig(os.path.join(d, "plots", "evidence.png"), dpi=90)
                plt.close(fig)
            except Exception as e:
                meta["plot_error"] = repr(e)
            hashes = {}
            for root, _, fs in os.walk(d):
                for fn in fs:
                    if fn == "hashes.json":
                        continue
                    pth = os.path.join(root, fn)
                    with open(pth, "rb") as f:
                        hashes[os.path.relpath(pth, d)] = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.join(d, "hashes.json"), "w") as f:
                json.dump(hashes, f, indent=1)
            meta["path"] = d
        idx = {k: meta[k] for k in ("id", "level", "level_name", "kind", "candidate", "tick", "ts", "claim",
                                    "replications", "passed_replications")}
        idx["path"] = meta.get("path")
        idx["effects"] = meta["effects"]; idx["p_values"] = meta["p_values"]
        self.discoveries.append(idx)
        return meta

    def load_discovery(self, did: str) -> Optional[Dict[str, Any]]:
        if self.paths is None or not re.fullmatch(r"D\d{5}-[A-Z0-9]+-[0-9a-f]{6}", did or ""):
            return None
        d = os.path.join(self.paths.discoveries, did)
        if not os.path.isdir(d):
            return None
        out = json.load(open(os.path.join(d, "metadata.json")))
        out["genome"] = np.load(os.path.join(d, "genome.npy"), allow_pickle=False)
        rp = os.path.join(d, "replay.npz")
        if os.path.exists(rp):
            z = np.load(rp, allow_pickle=False)
            out["replay"] = {"pos": z["pos"], "arena_of": z["arena_of"], "size": int(z["size"]), "ticks": int(z["ticks"]),
                             "layout": json.load(open(os.path.join(d, "replay_layout.json")))}
        return out

    def verify_discovery(self, did: str) -> Dict[str, Any]:
        d = os.path.join(self.paths.discoveries, did) if self.paths else ""
        if not d or not os.path.isdir(d):
            return {"ok": False, "reason": "not found"}
        hashes = json.load(open(os.path.join(d, "hashes.json")))
        bad = []
        for rel, h in hashes.items():
            with open(os.path.join(d, rel), "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != h:
                    bad.append(rel)
        return {"ok": not bad, "corrupted": bad}

    # ------------------------------------------------------------------ Pantheon (multi-eixo, revogável)
    def pantheon_consider(self, cand: Candidate, discovery_id: str):
        changed = []
        for axis, codes in PANTHEON_AXES.items():
            score = sum(LEVEL_WEIGHT.get(c, 1) * min(1.0, abs(cand.passed[c].get("effect", 0)) / 2 + 0.25)
                        for c in codes if c in cand.passed)
            if score <= 0:
                continue
            cur = self.pantheon.get(axis)
            if cur is None or score > cur["score"] or cur.get("revoked"):
                self.pantheon[axis] = {"axis": axis, "candidate": cand.cid, "score": score, "discovery": discovery_id,
                                       "levels": [c for c in codes if c in cand.passed], "ts": iso_now(),
                                       "versions_hash": ENGINE_VERSIONS_HASH, "revoked": False}
                changed.append(axis)
        held = collections.Counter(v["candidate"] for v in self.pantheon.values() if not v.get("revoked"))
        if held:
            cid, k = held.most_common(1)[0]
            if k >= 4 and (self.overall_champion is None or self.overall_champion["candidate"] != cid):
                self.overall_champion = {"candidate": cid, "axes": k, "ts": iso_now(),
                                         "rule": "holds ≥4 pantheon axes simultaneously (dominance across axes)"}
                changed.append("overall")
        return changed

    def revoke(self, axis: str, reason: str):
        if axis in self.pantheon:
            self.pantheon[axis]["revoked"] = True
            self.pantheon[axis]["revoked_reason"] = reason

    def state_dict(self):
        return {"zoo": [{**{k: v for k, v in asdict(c).items() if k != "genome"}, "genome": c.genome} for c in self.zoo.values()],
                "failures": list(self.failures), "weird": list(self.weird), "pantheon": self.pantheon,
                "discoveries": self.discoveries, "convergence": self.convergence, "overall": self.overall_champion}

    def load_state_dict(self, s):
        self.zoo = collections.OrderedDict()
        for d in s["zoo"]:
            d = dict(d); d["genome"] = np.asarray(d["genome"], np.float32)
            self.zoo[d["cid"]] = Candidate(**{k: v for k, v in d.items() if k in Candidate.__dataclass_fields__})
        self.failures = collections.deque(s["failures"], maxlen=5000); self.weird = collections.deque(s["weird"], maxlen=500)
        self.pantheon = dict(s["pantheon"]); self.discoveries = list(s["discoveries"])
        self.convergence = list(s.get("convergence", [])); self.overall_champion = s.get("overall")


# ----------------------------------------------------------------------------- DSL segura de experimentos
DSL_OPS = {"perturb", "freeze", "ablate", "duplicate", "fork", "change_environment", "observe", "measure"}
DSL_ENUMS = {
    "perturb": {"what": {"sensor_noise", "resource_scale", "body_gain", "initial_energy"}},
    "freeze": {"what": {"plasticity", "movement"}},
    "ablate": {"what": {"action", "sensor_group", "units"}},
    "change_environment": {"param": {"storm", "cold", "n_patches", "resource_floor", "rich", "water_barrier", "deposit"}},
    "measure": {"metric": {"intake", "survival", "distance", "placed", "crossed", "emission"}},
}
SENSOR_GROUPS = {"resource": SENSOR_RESOURCE, "intero": SENSOR_INTERO, "sound": SENSOR_SOUND, "extero": SENSOR_EXTERO}


def validate_dsl(prog: Any) -> Dict[str, Any]:
    """Valida um programa da DSL (dict JSON). Rejeita qualquer coisa fora do esquema — sem eval, sem exec."""
    if not isinstance(prog, dict):
        raise ValueError("programa deve ser um objeto")
    allowed_top = {"name", "target", "steps", "seed", "arenas"}
    if set(prog) - allowed_top:
        raise ValueError(f"campos não permitidos: {sorted(set(prog) - allowed_top)}")
    name = str(prog.get("name", "experiment"))[:80]
    if not re.fullmatch(r"[\w\- .]{1,80}", name):
        raise ValueError("nome inválido")
    tgt = prog.get("target", {})
    if not isinstance(tgt, dict) or tgt.get("kind") not in ("candidate", "world_fork"):
        raise ValueError("target.kind deve ser candidate|world_fork")
    if tgt.get("kind") == "candidate" and not re.fullmatch(r"[\w\-]{1,64}", str(tgt.get("id", ""))):
        raise ValueError("target.id inválido")
    steps = prog.get("steps", [])
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError("steps: 1–16 operações")
    clean = []
    total_ticks = 0
    for st in steps:
        if not isinstance(st, dict) or st.get("op") not in DSL_OPS:
            raise ValueError(f"operação não permitida: {st!r}"[:200])
        op = st["op"]
        c = {"op": op}
        for k, v in st.items():
            if k == "op":
                continue
            if k in ("what", "param", "metric", "channel", "group"):
                if not isinstance(v, str) or not re.fullmatch(r"[a-z_0-9]{1,32}", v):
                    raise ValueError(f"valor inválido em {k}")
                if op in DSL_ENUMS and k in DSL_ENUMS[op] and v not in DSL_ENUMS[op][k]:
                    raise ValueError(f"{op}.{k}={v} fora do vocabulário")
                if k == "channel" and v not in ACTION_NAMES:
                    raise ValueError("canal de ação desconhecido")
                if k == "group" and v not in SENSOR_GROUPS:
                    raise ValueError("grupo de sensores desconhecido")
                c[k] = v
            elif k in ("amount", "value"):
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or abs(v) > 10:
                    raise ValueError(f"{k} deve ser número finito |x|≤10")
                c[k] = float(v)
            elif k in ("ticks", "n"):
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 2000:
                    raise ValueError(f"{k} deve ser inteiro 1–2000")
                c[k] = int(v)
                if k == "ticks":
                    total_ticks += v
            else:
                raise ValueError(f"campo não permitido: {k}")
        clean.append(c)
    if total_ticks > 4000:
        raise ValueError("orçamento de ticks excedido (≤4000)")
    seed = prog.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 31:
        raise ValueError("seed inválida")
    ar = prog.get("arenas", 8)
    if not isinstance(ar, int) or isinstance(ar, bool) or not 1 <= ar <= 64:
        raise ValueError("arenas: 1–64")
    return {"name": name, "target": {"kind": tgt["kind"], "id": str(tgt.get("id", ""))}, "steps": clean, "seed": seed,
            "arenas": ar}


class ScientistEngine:
    PHASES = ["EXPLORE", "FIND", "INVESTIGATE", "VALIDATE"]

    def __init__(self, cfg: CivConfig, layout: GenomeLayout, validation: ValidationEngine, archive: ArchiveEngine,
                 qd: QDEvolution, logger: Optional[EventLogger], device: torch.device):
        self.cfg, self.L, self.ve, self.archive, self.qd, self.logger, self.device = cfg, layout, validation, archive, qd, logger, device
        self.queue: List[Dict[str, Any]] = []
        self.phase = "EXPLORE"
        self.current: Optional[Dict[str, Any]] = None
        self._gen: Optional[Generator] = None
        self.completed = 0
        self.discovery_events: List[Dict[str, Any]] = []
        self.results_log: "collections.deque" = collections.deque(maxlen=400)
        self.best_level = {"H": -1, "A": -1}
        self.validated_any: Dict[str, bool] = {}
        self.fitness_watch: "collections.deque" = collections.deque(maxlen=50)
        self.dsl_results: "collections.deque" = collections.deque(maxlen=100)
        self.rng = np.random.default_rng(derive_seed("scientist", cfg.master_seed) % (2 ** 32))
        self.budget_fraction = cfg.experiment_time_fraction
        self.seq = 0

    # ------------------------------------------------------------------ candidatos
    def add_candidate(self, genome: np.ndarray, origin: str, tick: int, fitness: float = 0.0, novelty: float = 0.0,
                      importance: float = 0.0, island: int = -1) -> Candidate:
        gh = hashlib.sha256(genome.astype(np.float32).tobytes()).hexdigest()[:16]
        cid = f"C{gh}"
        c = self.archive.zoo.get(cid)
        if c is None:
            c = Candidate(cid=cid, genome=genome.astype(np.float32), origin=origin, first_seen=iso_now(), tick=tick,
                          fitness=fitness, novelty=novelty, importance=importance, island=island)
            self.archive.add_candidate(c)
            self._check_convergence(c)
        else:
            c.fitness, c.novelty = max(c.fitness, fitness), max(c.novelty, novelty)
            c.importance = max(c.importance, importance)
        return c

    def _check_convergence(self, c: Candidate):
        """Convergência evolutiva: arquiteturas muito parecidas surgindo independentemente em ilhas diferentes."""
        if c.island < 0:
            return
        g = torch.as_tensor(c.genome, device=self.device).unsqueeze(0)
        for o in list(self.archive.zoo.values())[-200:]:
            if o.cid == c.cid or o.island < 0 or o.island == c.island:
                continue
            dist = float(self.L.distance(g, torch.as_tensor(o.genome, device=self.device).unsqueeze(0)))
            if dist < 0.05:
                ev = {"a": c.cid, "b": o.cid, "islands": [c.island, o.island], "distance": dist, "ts": iso_now()}
                self.archive.convergence.append(ev)
                self.archive.convergence = self.archive.convergence[-200:]
                break

    def next_codes(self, c: Candidate) -> List[Tuple[str, bool]]:
        """Próximos testes: o primeiro nível H não validado (gated) + triagem exploratória dos 2 seguintes; escada A
        após A0 (só para agentes que forrageiam)."""
        out = []
        missing = [x for x in H_CODES if x not in c.passed]
        if missing:
            i = H_CODES.index(missing[0])
            out.append((missing[0], True))
            for x in H_CODES[i + 1:i + 3]:
                if x not in c.passed and x not in c.screened:
                    out.append((x, False))
        am = [x for x in A_CODES[:9] if x not in c.passed]
        if am and c.fitness > 0.6:
            out.append((am[0], True))
        return out

    def priority(self, c: Candidate, code: str, prereq: bool, kind: str) -> float:
        lvl = LEVEL_WEIGHT.get(code, 1)
        tries = c.attempts.get(code, 0)
        unc = 1.0 / (1 + tries)
        voi = 1.5 if code == self.target_level() else 1.0
        base = {"screen": 1.0, "replicate": 3.0, "battery": 2.0, "dsl": 1.2}[kind]
        return float(base * (10 + 2 * lvl) * voi * unc * (1.4 if prereq else 0.6) + 3 * c.novelty + 2 * c.importance
                     + 0.5 * c.evidence_score())

    def target_level(self) -> str:
        best = max(self.archive.zoo.values(), key=lambda c: (c.h_level(), c.evidence_score()), default=None)
        lvl = best.h_level() if best else -1
        return H_CODES[min(len(H_CODES) - 1, lvl + 1)]

    def enqueue(self, c: Candidate, code: str, kind: str = "screen", prereq: bool = True, rep: int = 0):
        key = (c.cid, code, kind, rep)
        if any((q["cid"], q["code"], q["kind"], q["rep"]) == key for q in self.queue):
            return
        if kind == "screen" and c.attempts.get(code, 0) >= 3:
            return                                               # beco sem saída conhecido: arquivo de falhas evita repetição
        self.seq += 1
        self.queue.append({"cid": c.cid, "code": code, "kind": kind, "prereq": prereq, "rep": rep,
                           "created": time.time(), "prio": self.priority(c, code, prereq, kind), "seq": self.seq})
        if len(self.queue) > self.cfg.max_queue:
            self.queue.sort(key=lambda q: -q["prio"])
            self.queue = self.queue[: self.cfg.max_queue]        # descarta redundantes de baixo valor

    def plan(self):
        """Preenche a fila a partir do zoológico (candidatos mais promissores primeiro)."""
        cands = sorted(self.archive.zoo.values(), key=lambda c: -(c.fitness + c.novelty + c.evidence_score() + c.importance))[:40]
        for c in cands:
            for code, prereq in self.next_codes(c):
                self.enqueue(c, code, "screen", prereq)
        f = self.target_level()
        self.qd.set_focus(f)                                     # busca dirigida: evolução mira o próximo nível

    # ------------------------------------------------------------------ execução (gerador por fatias)
    def step(self, budget_s: float) -> Optional[Dict[str, Any]]:
        """Avança a ciência por até `budget_s` segundos. Devolve um relatório quando um experimento termina."""
        t0 = time.time()
        report = None
        while time.time() - t0 < budget_s:
            if self._gen is None:
                if not self.queue:
                    self.plan()
                if not self.queue:
                    self.phase = "EXPLORE"
                    return report
                for q in self.queue:
                    q["eff"] = q["prio"] * (1 + 0.02 * (time.time() - q["created"]))   # envelhecimento da fila
                self.queue.sort(key=lambda q: -q["eff"])
                self.current = self.queue.pop(0)
                kinds = {q["kind"] for q in self.queue} | {self.current["kind"]}
                self.phase = "VALIDATE" if "battery" in kinds else "INVESTIGATE" if "replicate" in kinds else "FIND"
                c = self.archive.zoo.get(self.current["cid"])
                if c is None:
                    self.current = None
                    continue
                key = c.cid if self.current["rep"] == 0 else f"{c.cid}|rep{self.current['rep']}"
                G = torch.as_tensor(c.genome, device=self.device)
                self._gen = self.ve.run_gen(self.current["code"], G, key)
                self.current["started"] = time.time()
            try:
                next(self._gen)
            except StopIteration as e:
                res: ProbeResult = e.value
                self._gen = None
                report = self._handle(self.current, res)
                self.current = None
        return report

    def _handle(self, exp: Dict[str, Any], res: ProbeResult) -> Dict[str, Any]:
        c = self.archive.zoo.get(exp["cid"])
        self.completed += 1
        code = exp["code"]
        rec = {"cid": exp["cid"], "code": code, "kind": exp["kind"], "rep": exp["rep"], "passed": res.passed, "p": res.p,
               "effect": res.effect, "duration": res.duration, "ts": res.ts, "claim": res.claim}
        self.results_log.append(rec)
        if self.logger:
            self.logger.event("probe", **{("exp_kind" if k == "kind" else k): v for k, v in rec.items()})
        if c is None:
            return rec
        c.attempts[code] = c.attempts.get(code, 0) + 1
        R = max(2, min(3, self.cfg.probe.replication_seeds))
        if exp["kind"] == "screen":
            if not res.passed:
                self.archive.failure(c.cid, res)
                c.failed[code] = c.failed.get(code, 0) + 1
                return rec
            c.screened[code] = {"p": res.p, "effect": res.effect, "reps": [res], "rep_pass": 0, "rep_fail": 0}
            for r in range(1, R + 1):
                self.enqueue(c, code, "replicate", True, rep=r)  # tentar quebrar em sementes de holdout independentes
            rec["next"] = f"replicate×{R}"
            return rec
        if exp["kind"] == "replicate":
            sc = c.screened.setdefault(code, {"reps": [], "rep_pass": 0, "rep_fail": 0})
            if res.passed:
                sc["rep_pass"] = sc.get("rep_pass", 0) + 1
                sc.setdefault("reps", []).append(res)
            else:
                sc["rep_fail"] = sc.get("rep_fail", 0) + 1
                self.archive.failure(c.cid, res)
            if sc.get("rep_pass", 0) + sc.get("rep_fail", 0) >= R:
                total_pass, total = 1 + sc.get("rep_pass", 0), R + 1
                if total_pass >= math.ceil(0.67 * total) and code not in c.passed:
                    self._discovery(c, code, sc.get("reps", []) or [res])
                    rec["discovery"] = True
                elif code not in c.passed:
                    c.failed[code] = c.failed.get(code, 0) + 1
                    rec["replication_failed"] = True
                sc["rep_pass"], sc["rep_fail"] = 0, 0
        return rec

    def _discovery(self, c: Candidate, code: str, results: List[ProbeResult]):
        c.passed[code] = {"effect": float(np.mean([r.effect for r in results if np.isfinite(r.effect)] or [0])),
                          "p": float(max(r.p for r in results if np.isfinite(r.p))) if any(np.isfinite(r.p) for r in results) else float("nan"),
                          "claim": results[0].claim, "ts": iso_now()}
        meta = self.archive.save_discovery(code, c, results, self.cfg, c.tick)
        c.passed[code]["discovery"] = meta["id"]
        self.validated_any[code] = True
        changed = self.archive.pantheon_consider(c, meta["id"])
        ev = {"type": "discovery", "id": meta["id"], "level": code, "level_name": LEVEL_NAME.get(code, code),
              "candidate": c.cid, "pantheon": changed, "claim": meta["claim"], "ts": iso_now()}
        self.discovery_events.append(ev)
        if self.logger:
            self.logger.event("discovery", echo=True, **{k: v for k, v in ev.items() if k != "type"})
        hl = max((x.h_level() for x in self.archive.zoo.values()), default=-1)
        al = max((x.a_level() for x in self.archive.zoo.values()), default=-1)
        self.best_level = {"H": hl, "A": al}
        if code.startswith("A") and all(x in c.passed or x in c.failed for x in A_CODES[1:9]):
            r9 = self.ve.affect.composite_A9(torch.as_tensor(c.genome, device=self.device))
            if r9.passed and "A9" not in c.passed:
                self._discovery(c, "A9", [r9])

    def record_civilization(self, level_name: str, tick: int, evidence: Dict[str, Any]):
        """Descoberta civilizacional (nível C subiu): pacote com evidência multidimensional do mundo."""
        c = Candidate(cid=f"WORLD-{level_name.split()[0]}", genome=np.zeros(1, np.float32), origin="world", first_seen=iso_now(), tick=tick)
        res = ProbeResult(level=level_name.split()[0], name=level_name, passed=True, p=float("nan"), effect=float("nan"),
                          ci=(float("nan"), float("nan")), n=1, metrics=to_jsonable(evidence), controls={
                              "rule": "multidimensional criteria held for ≥3 consecutive civilization windows"},
                          seeds=[], genome_hash="world", claim="SIMULATION HYPOTHESIS", duration=0.0,
                          thresholds=self.cfg.thresholds_hash())
        meta = self.archive.save_discovery(level_name.split()[0], c, [res], self.cfg, tick, kind="civilization")
        ev = {"type": "discovery", "id": meta["id"], "level": level_name.split()[0], "level_name": level_name,
              "candidate": "world", "claim": "SIMULATION HYPOTHESIS", "ts": iso_now()}
        self.discovery_events.append(ev)
        return meta

    def watch_fitness(self, best_fitness: float):
        """Anti-metric-gaming: fitness subindo muito sem nenhuma evidência nova → registrar anomalia."""
        self.fitness_watch.append((best_fitness, sum(len(c.passed) for c in self.archive.zoo.values())))
        if len(self.fitness_watch) == self.fitness_watch.maxlen:
            f0, e0 = self.fitness_watch[0]; f1, e1 = self.fitness_watch[-1]
            if f1 > (1 + self.cfg.thresholds.metric_exploit_gain) * max(1e-6, f0) and e1 == e0:
                self.archive.anomaly("fitness_without_evidence", {"fitness_from": f0, "fitness_to": f1, "evidence": e1})
                self.fitness_watch.clear()

    def adapt_budget(self) -> float:
        """Fila cresce → mais tempo para validação; fila pequena → mais exploração."""
        q = len(self.queue) / max(1, self.cfg.max_queue)
        self.budget_fraction = float(np.clip(self.cfg.experiment_time_fraction * (0.6 + 2.0 * q), 0.1, 0.6))
        return self.budget_fraction

    # ------------------------------------------------------------------ DSL
    def dsl_gen(self, prog: Dict[str, Any], runner: ArenaRunner):
        """Executa um programa validado em arenas de laboratório (alvo = candidato). Nunca toca o mundo principal."""
        p = validate_dsl(prog)
        if p["target"]["kind"] != "candidate":
            raise ValueError("world_fork é executado pelo UniverseEngine.counterfactual (fork marcado)")
        c = self.archive.zoo.get(p["target"]["id"])
        if c is None:
            raise ValueError("candidato desconhecido")
        n = p["arenas"]
        spec = ArenaSpec(task="dsl", n_patches=3, resource_floor=0.05)
        G = torch.as_tensor(c.genome, device=self.device).expand(n, -1).contiguous()
        mask = torch.ones(n, N_SENSORS, device=self.device)
        block = torch.zeros(n, N_ACTIONS, dtype=torch.bool, device=self.device)
        zero_units = torch.zeros(n, self.cfg.n_units, dtype=torch.bool, device=self.device)
        frz = mov = False
        noise, energy, body_gain = 0.0, 0.8, 1.0
        results, trial, rec = {}, None, None
        for st in p["steps"]:
            op = st["op"]
            if op == "perturb":
                if st["what"] == "sensor_noise":
                    noise = abs(st.get("amount", 0.1))
                elif st["what"] == "resource_scale":
                    spec.patch_value = float(np.clip(st.get("amount", 1.0), 0.0, 3.0))
                elif st["what"] == "initial_energy":
                    energy = float(np.clip(st.get("amount", 0.8), 0.05, 1.0))
                elif st["what"] == "body_gain":
                    body_gain = float(np.clip(st.get("amount", 1.0), 0.0, 3.0))
            elif op == "freeze":
                frz = frz or st["what"] == "plasticity"; mov = mov or st["what"] == "movement"
            elif op == "ablate":
                if st["what"] == "action" and "channel" in st:
                    block[:, ACTION_NAMES.index(st["channel"])] = True
                elif st["what"] == "sensor_group" and "group" in st:
                    mask[:, SENSOR_GROUPS[st["group"]]] = 0
                elif st["what"] == "units":
                    k = int(np.clip(st.get("n", 1), 1, self.cfg.n_units))
                    zero_units[:, :k] = True
            elif op == "duplicate":
                results["duplicates"] = int(st.get("n", 1))
            elif op == "change_environment":
                prm, val = st["param"], st.get("value", 0.0)
                if prm in ("storm", "cold", "resource_floor", "deposit"):
                    setattr(spec, prm, float(np.clip(val, 0.0, 3.0)))
                elif prm in ("n_patches", "rich"):
                    setattr(spec, prm, int(np.clip(val, 0, 8)))
                elif prm == "water_barrier":
                    spec.water_barrier = bool(val > 0); spec.spawn_side = "left" if val > 0 else "center"
            elif op == "fork":
                results.setdefault("forks", 0)
                results["forks"] += 1
            elif op == "observe":
                cap = runner.capability(G)
                cap = cap.clone(); cap[:, 0] *= body_gain
                trial = runner.build(G, spec, p["seed"], body_cap=cap, energy=energy)
                gen = torch.Generator(device=self.device); gen.manual_seed(p["seed"] + 1)

                def ctrl(t, tr, noise=noise):
                    add = torch.randn(tr.n, N_SENSORS, generator=gen, device=self.device) * noise if noise > 0 else None
                    iv = MindIntervention(sensor_mask=mask, sensor_add=add, zero_units=zero_units if zero_units.any() else None,
                                          freeze_plasticity=torch.full((tr.n,), frz, device=self.device) if frz else None)
                    return ArenaControl(iv=iv, action_block=block if block.any() else None,
                                        freeze_move=torch.ones(tr.n, dtype=torch.bool, device=self.device) if mov else None)
                rec = yield from runner.run_gen(trial, st.get("ticks", 100), ctrl_fn=ctrl,
                                                record=("gain", "alive", "speed", "placed", "emit"), name=f"DSL {p['name']}")
            elif op == "measure":
                if rec is None:
                    raise ValueError("measure antes de observe")
                m = st["metric"]
                val = {"intake": rec["gain"].sum(0), "survival": rec["alive"].float().mean(0), "distance": rec["speed"].sum(0),
                       "placed": rec["placed"].sum(0), "emission": rec["emit"].abs().sum(-1).mean(0),
                       "crossed": ArenaRunner.crossed(trial).float()}[m].cpu().numpy()
                results[m] = summary_stats(val)
        out = {"program": p, "results": results, "ts": iso_now(), "intervention": True, "scope": "lab arenas only"}
        self.dsl_results.append(out)
        return out

    # ------------------------------------------------------------------ estado
    def status(self) -> Dict[str, Any]:
        return {"phase": self.phase, "queue": len(self.queue), "completed": self.completed,
                "current": {k: v for k, v in (self.current or {}).items() if k in ("cid", "code", "kind", "rep")},
                "best_level": self.best_level, "target_level": self.target_level(), "budget_fraction": self.budget_fraction,
                "discoveries": len(self.archive.discoveries), "validated": sorted(self.validated_any.keys()),
                "recent": list(self.results_log)[-12:]}

    def state_dict(self):
        return {"queue": [{k: v for k, v in q.items() if k not in ("eff",)} for q in self.queue], "phase": self.phase,
                "completed": self.completed, "results_log": list(self.results_log), "best_level": self.best_level,
                "validated": self.validated_any, "seq": self.seq, "rng": self.rng.bit_generator.state,
                "screened": {cid: {code: {"p": v.get("p"), "effect": v.get("effect"), "rep_fail": v.get("rep_fail", 0),
                                          "rep_pass": v.get("rep_pass", 0)}
                                   for code, v in c.screened.items()} for cid, c in self.archive.zoo.items() if c.screened},
                "current": self.current}

    def load_state_dict(self, s):
        now = time.time()
        self.queue = [dict(q, created=now) for q in s["queue"]]
        if s.get("current"):                                         # experimento interrompido volta para a fila
            self.queue.insert(0, dict(s["current"], created=now))
        self.phase, self.completed = s["phase"], int(s["completed"])
        self.results_log = collections.deque(s["results_log"], maxlen=400)
        self.best_level, self.validated_any, self.seq = dict(s["best_level"]), dict(s["validated"]), int(s["seq"])
        try:
            self.rng.bit_generator.state = s["rng"]
        except Exception:
            pass
        for cid, sc in s.get("screened", {}).items():
            c = self.archive.zoo.get(cid)
            if c is not None:
                for code, v in sc.items():
                    c.screened[code] = {"p": v.get("p"), "effect": v.get("effect"), "rep_fail": v.get("rep_fail", 0),
                                        "rep_pass": v.get("rep_pass", 0), "reps": []}
