# %% [markdown]
# ## 13 — EVOLUTION / QD: ilhas evolutivas, MAP-Elites, novidade, Pareto, espécies, bandit de operadores
# Duas camadas evolutivas (nunca confundidas com aprendizado individual):
# 1. **Evolução no mundo**: reprodução com mutação/recombinação dentro do planeta (seleção natural, sem fitness
#    explícito — módulo 10).
# 2. **Ilhas de laboratório** (aqui): populações independentes avaliadas nas arenas com **vetor de capacidades**
#    (sobrevivência, ingestão, eficiência, deslocamento, construção, travessia…), descritor comportamental separado
#    (MAP-Elites 3D + arquivo de novidade) e frente de Pareto. Migração rara entre ilhas; imigrantes raros para o
#    planeta (origem registrada na linhagem: `immigrant_qd`).
# **Busca dirigida**: o Scientist informa o próximo nível ainda não validado; uma ilha-foco passa a avaliar a tarefa
# desse nível com um proxy barato (o proxy só orienta a busca — a validação continua no módulo 14, com controles).
# Fitness ≠ evidência: os dois são mostrados separadamente.

# %%
QD_OPERATORS = ["small", "medium", "large", "crossover", "elite_cross"]
FOCUS_TASKS = {  # nível alvo → (spec da arena, agentes por arena, proxy)
    "H0": (ArenaSpec(task="forage"), 1, "develop"),
    "H1": (ArenaSpec(task="forage"), 1, "locomotion"),
    "H2": (ArenaSpec(task="forage", n_patches=4, resource_floor=0.05), 1, "intake"),
    "H3": (ArenaSpec(task="forage", n_patches=2, regen=True), 1, "improvement"),
    "H4": (ArenaSpec(task="cue", hidden_side=True, patch_radius=1), 1, "cue"),
    "H5": (ArenaSpec(task="forage", n_patches=3), 1, "generalize"),
    "H6": (ArenaSpec(task="cue", hidden_side=True), 1, "cue"),
    "H7": (ArenaSpec(task="barrier", water_barrier=True, deposit=3.0, spawn_side="left", n_patches=3), 1, "crossing"),
    "H8": (ArenaSpec(task="coop", rich=2, n_patches=1), 2, "pair_intake"),
    "H9": (ArenaSpec(task="signal", n_patches=2), 2, "receiver"),
    "H10": (ArenaSpec(task="coop", rich=2, n_patches=2), 2, "pair_intake"),
    "H11": (ArenaSpec(task="coop", rich=2, n_patches=2, regen=True), 2, "pair_intake"),
    "H12": (ArenaSpec(task="forage", n_patches=3, storm=0.3), 1, "novelty"),
}
ISLAND_ENVS = [ArenaSpec(task="forage", n_patches=3, resource_floor=0.1),
               ArenaSpec(task="forage", n_patches=2, cold=0.6, deposit=2.0, resource_floor=0.1),
               ArenaSpec(task="barrier", water_barrier=True, deposit=3.0, spawn_side="left", n_patches=3, resource_floor=0.05),
               ArenaSpec(task="coop", rich=2, n_patches=1, resource_floor=0.1),
               ArenaSpec(task="forage", n_patches=4, storm=0.4, deposit=2.0, resource_floor=0.1),
               ArenaSpec(task="forage", n_patches=2, regen=True, resource_floor=0.05), ArenaSpec(task="cue", hidden_side=True),
               ArenaSpec(task="forage", n_patches=1, patch_radius=2, resource_floor=0.1)]


class MAPElites:
    def __init__(self, bins: Sequence[int], P: int):
        self.bins = list(bins)
        self.P = P
        self.cells: Dict[int, Dict[str, Any]] = {}

    def cell_of(self, desc: np.ndarray) -> np.ndarray:
        idx = np.zeros(desc.shape[0], np.int64)
        for j, b in enumerate(self.bins):
            idx = idx * b + np.clip((desc[:, j] * b).astype(int), 0, b - 1)
        return idx

    def insert(self, G: np.ndarray, fit: np.ndarray, desc: np.ndarray, qvec: np.ndarray, gids: Sequence[str], gen: int) -> int:
        cells = self.cell_of(desc)
        new = 0
        for i, c in enumerate(cells.tolist()):
            cur = self.cells.get(c)
            if cur is None or fit[i] > cur["fit"]:
                if cur is None:
                    new += 1
                self.cells[c] = {"fit": float(fit[i]), "genome": G[i].astype(np.float32), "desc": desc[i].tolist(),
                                 "qvec": qvec[i].tolist(), "id": gids[i], "gen": gen}
        return new

    def sample(self, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not self.cells or n <= 0:
            return None
        keys = list(self.cells.keys())
        pick = rng.integers(0, len(keys), n)
        return np.stack([self.cells[keys[k]]["genome"] for k in pick])

    def coverage(self) -> float:
        return len(self.cells) / float(np.prod(self.bins))

    def qd_score(self) -> float:
        return float(sum(max(0.0, c["fit"]) for c in self.cells.values()))

    def state_dict(self):
        keys = sorted(self.cells.keys())
        return {"bins": self.bins, "keys": np.asarray(keys, np.int64),
                "genomes": np.stack([self.cells[k]["genome"] for k in keys]) if keys else np.zeros((0, self.P), np.float32),
                "meta": [{k2: v for k2, v in self.cells[k].items() if k2 != "genome"} for k in keys]}

    def load_state_dict(self, s):
        self.bins = list(s["bins"])
        self.cells = {}
        for k, g, m in zip(np.asarray(s["keys"]).tolist(), np.asarray(s["genomes"]), s["meta"]):
            self.cells[int(k)] = dict(m, genome=np.asarray(g, np.float32))


class NoveltyArchive:
    def __init__(self, cap: int = 2000, k: int = 8):
        self.cap, self.k = cap, k
        self.items = np.zeros((0, 8), np.float32)
        self.seen = 0

    def novelty(self, B: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        ref = self.items if extra is None else np.concatenate([self.items, extra])
        if ref.shape[0] < 2:
            return np.ones(B.shape[0], np.float32)
        d = np.sqrt(((B[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
        d.sort(1)
        k = min(self.k, d.shape[1] - 1)
        return d[:, 1:k + 1].mean(1).astype(np.float32)

    def add(self, B: np.ndarray, rng: np.random.Generator):
        for b in B:
            self.seen += 1
            if self.items.shape[0] < self.cap:
                self.items = np.concatenate([self.items, b[None].astype(np.float32)])
            else:
                j = int(rng.integers(0, self.seen))
                if j < self.cap:
                    self.items[j] = b


class OperatorBandit:
    def __init__(self, ops: Sequence[str]):
        self.ops = list(ops)
        self.n = np.ones(len(ops)); self.r = np.ones(len(ops)) * 0.5
        self.t = 1

    def choose(self, rng: np.random.Generator, k: int) -> np.ndarray:
        ucb = self.r / self.n + np.sqrt(2 * np.log(self.t + 1) / self.n)
        p = np.exp((ucb - ucb.max()) * 6); p /= p.sum()
        return rng.choice(len(self.ops), size=k, p=p)

    def update(self, ops: np.ndarray, rewards: np.ndarray):
        for o, r in zip(ops.tolist(), rewards.tolist()):
            self.n[o] += 1; self.r[o] += float(r)
        self.t += len(ops)

    def stats(self) -> Dict[str, float]:
        return {op: float(self.r[i] / self.n[i]) for i, op in enumerate(self.ops)}


def pareto_front(Q: np.ndarray) -> np.ndarray:
    """Índices não dominados (maximização em todas as colunas)."""
    n = Q.shape[0]
    dom = np.zeros(n, bool)
    for i in range(n):
        if dom[i]:
            continue
        better_eq = (Q >= Q[i]).all(1) & (Q > Q[i]).any(1)
        if better_eq.any():
            dom[i] = True
    return np.nonzero(~dom)[0]


@dataclass
class Island:
    iid: int
    spec: ArenaSpec
    per_arena: int
    pop: np.ndarray                      # [n, P] float32
    fit: np.ndarray
    qvec: np.ndarray
    desc: np.ndarray
    gids: List[str]
    generation: int = 0
    best: float = -1e9
    stagnant: int = 0
    sigma_mult: float = 1.0
    novelty_w: float = 0.3
    focus: str = ""
    history: List[float] = field(default_factory=list)


QVEC_NAMES = ["survival", "intake", "efficiency", "distance", "building", "crossing", "final_energy", "learning"]


class QDEvolution:
    def __init__(self, cfg: CivConfig, layout: GenomeLayout, runner: ArenaRunner, device: torch.device,
                 logger: Optional[EventLogger] = None):
        self.cfg, self.L, self.runner, self.device, self.logger = cfg, layout, runner, device, logger
        self.map = MAPElites(cfg.map_bins, layout.P)
        self.novelty = NoveltyArchive()
        self.bandit = OperatorBandit(QD_OPERATORS)
        self.islands: List[Island] = []
        self.rng = np.random.default_rng(derive_seed("qd", cfg.master_seed) % (2 ** 32))
        self.gen = torch.Generator(device=device); self.gen.manual_seed(derive_seed("qd_torch", cfg.master_seed))
        self.focus_level = ""
        self.evaluations = 0
        self.generations = 0
        self.eval_seconds = 0.0
        self.last_report: Dict[str, Any] = {}
        self.events: "collections.deque" = collections.deque(maxlen=200)
        self._rr = 0

    # ------------------------------------------------------------------ população
    def init_islands(self):
        for i in range(self.cfg.n_islands):
            self._new_island(ISLAND_ENVS[i % len(ISLAND_ENVS)], 2 if ISLAND_ENVS[i % len(ISLAND_ENVS)].task == "coop" else 1)

    def _new_island(self, spec: ArenaSpec, per_arena: int) -> Island:
        n = self.cfg.island_pop
        G = self.L.random(n, self.gen).cpu().numpy().astype(np.float32)
        isl = Island(iid=len(self.islands), spec=copy.deepcopy(spec), per_arena=per_arena, pop=G,
                     fit=np.full(n, -1e9, np.float32), qvec=np.zeros((n, len(QVEC_NAMES)), np.float32),
                     desc=np.zeros((n, 3), np.float32), gids=[new_id("g") for _ in range(n)])
        self.islands.append(isl)
        return isl

    def set_focus(self, level: str):
        if level == self.focus_level or level not in FOCUS_TASKS:
            return
        self.focus_level = level
        if not self.islands:
            return
        isl = self.islands[0]
        spec, per, _ = FOCUS_TASKS[level]
        isl.spec, isl.per_arena, isl.focus = copy.deepcopy(spec), per, level
        isl.fit[:] = -1e9                          # a ilha-foco reavalia na nova tarefa
        isl.best, isl.stagnant = -1e9, 0
        self.events.append({"t": iso_now(), "type": "focus", "level": level})

    # ------------------------------------------------------------------ avaliação (gerador)
    def evaluate_gen(self, G: torch.Tensor, spec: ArenaSpec, per_arena: int, seed: int, proxy: str = "",
                     ticks: Optional[int] = None):
        """Avalia genomas em arena: devolve (fitness proxy, qvec, desc). Cada genoma ocupa `per_arena` agentes."""
        ticks = ticks or self.cfg.probe.arena_ticks
        n = G.shape[0]
        Gr = G.repeat_interleave(per_arena, 0)
        t0 = time.time()
        cap = self.runner.capability(G)
        capr = cap.repeat_interleave(per_arena, 0)
        tr = self.runner.build(Gr, spec, seed, per_arena=per_arena, body_cap=capr)
        ctrl_fn = None
        if spec.hidden_side:
            lab = tr.label[tr.arena_of]

            def ctrl_fn(t, trial, lab=lab):
                if t < 8:
                    add = torch.zeros(trial.n, N_SENSORS, device=self.device)
                    add[:, 12] = (lab == 0).float() * 1.5
                    add[:, 13] = (lab == 1).float() * 1.5
                    return ArenaControl(iv=MindIntervention(sensor_add=add))
                return None
        elif spec.task == "signal":
            vis = torch.zeros(tr.n, dtype=torch.bool, device=self.device)
            vis[0::2] = True                                        # emissor vê o recurso; receptor não

            def ctrl_fn(t, trial, vis=vis):
                return ArenaControl(resource_visible=vis)
        rec = yield from self.runner.run_gen(tr, ticks, ctrl_fn=ctrl_fn, record=("gain", "alive", "speed", "placed", "emit", "out"),
                                             name=f"QD {spec.task}")
        gain, alive = rec["gain"], rec["alive"].float()
        T = gain.shape[0]
        intake = gain.sum(0)
        surv = alive.mean(0)
        dist = rec["speed"].sum(0)
        placed = rec["placed"].sum(0)
        third = max(1, T // 3)
        learn = gain[-third:].sum(0) - gain[:third].sum(0)
        crossed = ArenaRunner.crossed(tr).float()
        emit = rec["emit"].abs().sum(-1).mean(0)
        spent = (0.8 - tr.pool.energy + intake).clamp(min=1e-3)
        per = lambda v: v.view(n, per_arena)
        q = torch.stack([per(surv).mean(1), per(intake).mean(1), per(intake / spent).mean(1), per(dist).mean(1) / T,
                         per(placed).sum(1), per(crossed).max(1).values, per(tr.pool.energy).mean(1),
                         per(learn).mean(1)], 1)
        if spec.task == "signal":
            q[:, 1] = per(intake)[:, 1]                             # ingestão do receptor cego
        desc = torch.stack([(cap[:, 2] / self.runner.dev.N).clamp(0, 1), (q[:, 3] / self.cfg.max_speed_cap).clamp(0, 1),
                            (q[:, 4] / 6).clamp(0, 1)], 1)
        # propensão reprodutiva (porta de reprodução aberta na 2ª metade): capacidade necessária para persistir no planeta
        repro = per((torch.sigmoid(rec["out"][T // 2:, :, 9]) > 0.5).float().mean(0)).mean(1)
        fit = q[:, 0] * 0.5 + q[:, 1] + 0.3 * repro
        if proxy == "develop":
            fit = fit + 0.5 * desc[:, 0] * (cap[:, 2] >= self.cfg.thresholds.dev_min_nodes).float()
        elif proxy == "locomotion":
            fit = fit + 10.0 * cap[:, 0]
        elif proxy == "improvement":
            fit = fit + 2.0 * q[:, 7].clamp(min=0)
        elif proxy in ("crossing",):
            fit = fit + 1.0 * q[:, 5] + 0.05 * q[:, 4].clamp(max=6)
        self.evaluations += n
        self.eval_seconds += time.time() - t0
        return fit.cpu().numpy(), q.cpu().numpy(), desc.cpu().numpy(), cap.cpu().numpy(), emit.view(n, per_arena).mean(1).cpu().numpy()

    def _offspring(self, isl: Island, k: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        d = self.device
        ops = self.bandit.choose(self.rng, k)
        valid = isl.fit > -1e8
        # torneio com fitness + novidade (diversidade)
        score = np.where(valid, isl.fit, -1e9)
        pidx = []
        for _ in range(k):
            a, b = self.rng.integers(0, len(score), 2)
            pidx.append(a if score[a] >= score[b] else b)
        pidx = np.asarray(pidx)
        P = torch.as_tensor(isl.pop[pidx], device=d)
        children = torch.empty_like(P)
        for oi, op in enumerate(QD_OPERATORS):
            m = np.nonzero(ops == oi)[0]
            if m.size == 0:
                continue
            mi = torch.as_tensor(m, device=d)
            par = P[mi]
            if op == "small":
                ch = self.L.mutate(par, self.gen, 0.5 * isl.sigma_mult)
            elif op == "medium":
                ch = self.L.mutate(par, self.gen, 1.0 * isl.sigma_mult)
            elif op == "large":
                ch = self.L.mutate(par, self.gen, 3.0 * isl.sigma_mult)
            elif op == "crossover":
                mates = torch.as_tensor(isl.pop[self.rng.integers(0, len(isl.pop), m.size)], device=d)
                ch = self.L.mutate(self.L.crossover(par, mates, self.gen), self.gen, 0.5 * isl.sigma_mult)
            else:
                el = self.map.sample(m.size, self.rng)
                mates = torch.as_tensor(el, device=d) if el is not None else par
                ch = self.L.mutate(self.L.crossover(par, mates, self.gen), self.gen, 0.5 * isl.sigma_mult)
            children[mi] = ch
        return children, ops, pidx

    def generation_gen(self):
        """Uma geração de uma ilha (round-robin). Gerador: rende durante a avaliação na arena."""
        if not self.islands:
            self.init_islands()
        isl = self.islands[self._rr % len(self.islands)]
        self._rr += 1
        proxy = FOCUS_TASKS[isl.focus][2] if isl.focus else ""
        seed = derive_seed("qd_eval", isl.iid, isl.generation)
        if (isl.fit <= -1e8).any():                  # avalia população inicial / reavaliação de foco
            idx = np.nonzero(isl.fit <= -1e8)[0][: self.cfg.qd_batch]
            G = torch.as_tensor(isl.pop[idx], device=self.device)
            f, q, dsc, cap, emit = yield from self.evaluate_gen(G, isl.spec, isl.per_arena, seed, proxy)
            isl.fit[idx], isl.qvec[idx], isl.desc[idx] = f, q, dsc
            self.map.insert(isl.pop[idx], f, dsc, q, [isl.gids[i] for i in idx], isl.generation)
            return {"island": isl.iid, "kind": "init", "n": len(idx)}
        k = self.cfg.qd_batch
        children, ops, pidx = self._offspring(isl, k)
        f, q, dsc, cap, emit = yield from self.evaluate_gen(children, isl.spec, isl.per_arena, seed, proxy)
        beh = np.concatenate([dsc, q[:, [0, 1, 5, 6]] / np.maximum(1e-6, np.abs(q[:, [0, 1, 5, 6]]).max(0)),
                              emit[:, None]], 1).astype(np.float32)
        nov = self.novelty.novelty(beh)
        self.novelty.add(beh[np.argsort(-nov)[: max(1, k // 6)]], self.rng)
        sel = f + isl.novelty_w * nov * (np.abs(f).mean() + 1e-3) * (1.0 if proxy != "novelty" else 4.0)
        rew = (f > isl.fit[pidx]).astype(np.float32)
        self.bandit.update(ops, rew)
        Gc = children.cpu().numpy().astype(np.float32)
        gids = [new_id("g") for _ in range(k)]
        newcells = self.map.insert(Gc, f, dsc, q, gids, isl.generation)
        # (μ+λ) com fitness de seleção (fitness + novidade), preservando a frente de Pareto
        allG = np.concatenate([isl.pop, Gc]); allF = np.concatenate([isl.fit, f]); allS = np.concatenate([isl.fit, sel])
        allQ = np.concatenate([isl.qvec, q]); allD = np.concatenate([isl.desc, dsc]); allI = isl.gids + gids
        front = pareto_front(allQ[:, [0, 1, 4, 5]])
        order = list(dict.fromkeys(list(front[np.argsort(-allS[front])][: self.cfg.island_pop // 3]) +
                                   list(np.argsort(-allS))))[: self.cfg.island_pop]
        isl.pop, isl.fit, isl.qvec, isl.desc = allG[order], allF[order], allQ[order], allD[order]
        isl.gids = [allI[i] for i in order]
        isl.generation += 1
        self.generations += 1
        best = float(isl.fit.max())
        isl.history.append(best)
        isl.history = isl.history[-200:]
        if best > isl.best + 1e-3:
            isl.best, isl.stagnant = best, 0
            isl.sigma_mult = max(0.5, isl.sigma_mult * 0.9)
        else:
            isl.stagnant += 1
            self._stagnation(isl)
        if len(self.islands) > 1 and self.rng.random() < 0.05:   # migração rara entre ilhas
            other = self.islands[int(self.rng.integers(0, len(self.islands)))]
            if other is not isl:
                j = int(np.argmax(isl.fit)); w = int(np.argmin(other.fit))
                other.pop[w], other.fit[w] = isl.pop[j].copy(), -1e9
        self.last_report = {"island": isl.iid, "task": isl.spec.task, "focus": isl.focus, "gen": isl.generation,
                            "best": best, "mean": float(np.mean(f)), "new_cells": newcells,
                            "coverage": self.map.coverage(), "qd_score": self.map.qd_score(),
                            "bandit": self.bandit.stats(), "stagnant": isl.stagnant}
        return self.last_report

    def _stagnation(self, isl: Island):
        """Resposta gradual à estagnação: mutação ↑ → novidade ↑ → imigrantes aleatórios → nova ilha → novo ambiente →
        reinício parcial (mantém elites)."""
        s = isl.stagnant
        step = None
        if s == 6:
            isl.sigma_mult = min(4.0, isl.sigma_mult * 2); step = "mutation_up"
        elif s == 10:
            isl.novelty_w = min(2.0, isl.novelty_w * 2); step = "novelty_up"
        elif s == 14:
            n = len(isl.pop) // 4
            isl.pop[-n:] = self.L.random(n, self.gen).cpu().numpy(); isl.fit[-n:] = -1e9; step = "random_immigrants"
        elif s == 18 and len(self.islands) < self.cfg.n_islands + 2:
            self._new_island(ISLAND_ENVS[len(self.islands) % len(ISLAND_ENVS)], 1); step = "new_island"
        elif s == 22 and not isl.focus:
            isl.spec = copy.deepcopy(ISLAND_ENVS[int(self.rng.integers(0, len(ISLAND_ENVS)))])
            isl.per_arena = 2 if isl.spec.task == "coop" else 1
            isl.fit[:] = -1e9; step = "new_environment"
        elif s >= 26:
            keep = max(2, len(isl.pop) // 6)
            order = np.argsort(-isl.fit)
            isl.pop = isl.pop[order]; isl.fit = isl.fit[order]
            isl.pop[keep:] = self.L.random(len(isl.pop) - keep, self.gen).cpu().numpy()
            isl.fit[keep:] = -1e9
            isl.stagnant, isl.sigma_mult, isl.novelty_w = 0, 1.0, 0.3
            step = "partial_restart"
        if step:
            ev = {"t": iso_now(), "type": "stagnation_response", "island": isl.iid, "step": step}
            self.events.append(ev)
            if self.logger:
                self.logger.event("qd_stagnation", **ev)

    # ------------------------------------------------------------------ acesso
    def best_genomes(self, k: int) -> Tuple[np.ndarray, List[str]]:
        pool = []
        for isl in self.islands:
            ok = isl.fit > -1e8
            for i in np.nonzero(ok)[0]:
                pool.append((float(isl.fit[i]), isl.pop[i], isl.gids[i]))
        pool.sort(key=lambda x: -x[0])
        sel = pool[:k]
        if not sel:
            return np.zeros((0, self.L.P), np.float32), []
        return np.stack([s[1] for s in sel]), [s[2] for s in sel]

    def elites_for_founding(self, k: int) -> np.ndarray:
        G, _ = self.best_genomes(k)
        if self.map.cells:
            extra = self.map.sample(max(0, k - len(G)), self.rng)
            if extra is not None and len(extra):
                G = np.concatenate([G, extra]) if len(G) else extra
        return G[:k]

    def summary(self) -> Dict[str, Any]:
        return {"islands": [{"id": i.iid, "task": i.spec.task, "focus": i.focus, "gen": i.generation,
                             "best": float(i.best) if i.best > -1e8 else None, "stagnant": i.stagnant,
                             "history": [float(x) for x in i.history[-60:]]} for i in self.islands],
                "coverage": self.map.coverage(), "qd_score": self.map.qd_score(), "cells": len(self.map.cells),
                "evaluations": self.evaluations, "generations": self.generations, "focus": self.focus_level,
                "bandit": self.bandit.stats(), "novelty_archive": int(self.novelty.items.shape[0]),
                "eval_rate": self.evaluations / max(1e-6, self.eval_seconds)}

    def state_dict(self):
        return {"map": self.map.state_dict(), "novelty": self.novelty.items, "novelty_seen": self.novelty.seen,
                "bandit_n": self.bandit.n, "bandit_r": self.bandit.r, "bandit_t": self.bandit.t,
                "focus": self.focus_level, "evaluations": self.evaluations, "generations": self.generations,
                "rr": self._rr, "rng": self.rng.bit_generator.state, "torch_rng": self.gen.get_state().cpu().numpy(),
                "islands": [{"iid": i.iid, "spec": asdict(i.spec), "per_arena": i.per_arena, "pop": i.pop, "fit": i.fit,
                             "qvec": i.qvec, "desc": i.desc, "gids": i.gids, "generation": i.generation,
                             "best": i.best, "stagnant": i.stagnant, "sigma_mult": i.sigma_mult,
                             "novelty_w": i.novelty_w, "focus": i.focus, "history": i.history} for i in self.islands]}

    def load_state_dict(self, s):
        self.map.load_state_dict(s["map"])
        self.novelty.items = np.asarray(s["novelty"], np.float32); self.novelty.seen = int(s["novelty_seen"])
        self.bandit.n, self.bandit.r, self.bandit.t = np.asarray(s["bandit_n"]), np.asarray(s["bandit_r"]), int(s["bandit_t"])
        self.focus_level = s["focus"]; self.evaluations = int(s["evaluations"]); self.generations = int(s["generations"])
        self._rr = int(s["rr"])
        try:
            self.rng.bit_generator.state = s["rng"]
            self.gen.set_state(torch.as_tensor(np.asarray(s["torch_rng"]), dtype=torch.uint8))
        except Exception as e:
            if self.logger:
                self.logger.event("qd_rng_restore_failed", error=repr(e))
        self.islands = []
        for d in s["islands"]:
            sp = ArenaSpec(**{k: v for k, v in d["spec"].items() if k in ArenaSpec.__dataclass_fields__})
            self.islands.append(Island(iid=int(d["iid"]), spec=sp, per_arena=int(d["per_arena"]),
                                       pop=np.asarray(d["pop"], np.float32), fit=np.asarray(d["fit"], np.float32),
                                       qvec=np.asarray(d["qvec"], np.float32), desc=np.asarray(d["desc"], np.float32),
                                       gids=list(d["gids"]), generation=int(d["generation"]), best=float(d["best"]),
                                       stagnant=int(d["stagnant"]), sigma_mult=float(d["sigma_mult"]),
                                       novelty_w=float(d["novelty_w"]), focus=d["focus"],
                                       history=[float(x) for x in d["history"]]))
