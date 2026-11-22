# %% [markdown]
# ## 7. Quality-Diversity e arquivos permanentes
# * **MAP-Elites** sobre eixos comportamentais (padrão: persistência × propagação causal × complexidade temporal).
#   Elites substituídas nunca são apagadas (histórico de elites).
# * **Novelty archive** (kNN no espaço de descritores) — um universo pode ser "ruim" e muito diferente: é preservado.
# * **Genesis Zoo** (categorias descritivas externas, múltiplas por universo; `UNKNOWN` e `ANOMALOUS` são cidadãos de primeira classe).
# * **Champion / Anomaly / Failure archives**, **linhagem** completa, **especiação** por genótipo + comportamento,
#   **modelo de surpresa** (z-score frente ao histórico), **bandit UCB** para operadores de mutação.

# %%
ZOO_CATEGORIES = ["STATIC", "OSCILLATORY", "PROPAGATING", "PERSISTENT", "MOBILE", "RECOVERING", "MEMORY_CANDIDATE",
                  "CHANNEL_CANDIDATE", "TRANSFORMER_CANDIDATE", "UNKNOWN", "ANOMALOUS"]


def pack_params(space: ParamSpace, params: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(params[n], np.float32).ravel() for n in space.specs]).astype(np.float32)


def unpack_params(space: ParamSpace, vec: np.ndarray) -> Dict[str, np.ndarray]:
    out, i = {}, 0
    for n, s in space.specs.items():
        k = int(np.prod(s.shape)) if s.shape else 1
        v = np.asarray(vec[i:i + k], np.float64).reshape(s.shape)
        i += k
        if s.kind == "bool":
            v = v > 0.5
        elif s.kind in ("int", "cat"):
            v = np.rint(v).astype(np.int64)
        else:
            v = v.astype(np.float32)
        out[n] = v
    return space.clamp(out)


def genome_to_record(space: ParamSpace, g: Genome) -> dict:
    d = g.to_dict(); d["packed"] = pack_params(space, g.params); del d["params"]
    return d


def genome_from_record(space: ParamSpace, d: dict) -> Genome:
    d = dict(d); d["params"] = unpack_params(space, d.pop("packed"))
    return Genome.from_dict(d)


class MapElites:
    def __init__(self, axes: List[str], bins: List[int]):
        self.axes, self.bins = list(axes), list(bins)
        self.cells: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.insertions = 0

    def index(self, d: Dict[str, float]) -> Tuple[int, ...]:
        return tuple(int(min(b - 1, max(0, math.floor(float(np.clip(d.get(a, 0.0), 0, 1)) * b))))
                     for a, b in zip(self.axes, self.bins))

    def try_insert(self, gid: str, d: Dict[str, float], quality: float, gen: int) -> Tuple[bool, bool]:
        key = self.index(d)
        cur = self.cells.get(key)
        if cur is None or quality > cur["quality"]:
            if cur is not None:
                self.history.append({**cur, "cell": list(key), "replaced_at": gen})
            self.cells[key] = {"genome_id": gid, "quality": float(quality), "generation": gen}
            self.insertions += 1
            return True, cur is None
        return False, False

    def coverage(self) -> float:
        return len(self.cells) / float(np.prod(self.bins))

    def heat(self) -> np.ndarray:
        H = np.full(self.bins[:2], np.nan)
        for k, v in self.cells.items():
            i, j = k[0], k[1]
            H[i, j] = v["quality"] if np.isnan(H[i, j]) else max(H[i, j], v["quality"])
        return H

    def ids(self) -> List[str]:
        return [v["genome_id"] for v in self.cells.values()]

    def sample(self, rng: np.random.Generator, k: int) -> List[str]:
        ids = self.ids()
        if not ids:
            return []
        return [ids[i] for i in rng.integers(0, len(ids), size=k)]

    def state_dict(self):
        return {"axes": self.axes, "bins": self.bins, "cells": {",".join(map(str, k)): v for k, v in self.cells.items()},
                "history": self.history, "insertions": self.insertions}

    def load_state_dict(self, s):
        self.axes, self.bins = s["axes"], s["bins"]
        self.cells = {tuple(int(x) for x in k.split(",")): v for k, v in s["cells"].items()}
        self.history, self.insertions = s["history"], s["insertions"]


class NoveltyArchive:
    def __init__(self, k: int, max_size: int, target_add_rate: float = 0.05):
        self.k, self.max_size = k, max_size
        self.vecs = np.zeros((0, len(DESCRIPTOR_NAMES)), np.float32)
        self.ids: List[str] = []
        self.threshold = 0.15
        self.target = target_add_rate
        self.recent_adds: List[int] = []

    def novelty(self, V: np.ndarray, others: Optional[np.ndarray] = None) -> np.ndarray:
        ref = self.vecs if others is None else (np.vstack([self.vecs, others]) if len(self.vecs) else others)
        if len(ref) == 0:
            return np.ones(len(V), np.float32)
        Vt, Rt = torch.as_tensor(np.asarray(V, np.float32)), torch.as_tensor(np.asarray(ref, np.float32))
        d = torch.cdist(Vt, Rt)
        d = torch.where(d < 1e-9, torch.full_like(d, 1e6), d)     # exclui a auto-correspondência
        k = min(self.k, d.shape[1])
        nn = torch.topk(d, k, dim=1, largest=False).values
        valid = (nn < 1e6).float()
        out = (nn * valid).sum(1) / valid.sum(1).clamp(min=1)
        out = torch.where(valid.sum(1) > 0, out, torch.ones_like(out))
        return out.numpy().astype(np.float32)

    def maybe_add(self, ids: List[str], V: np.ndarray, nov: np.ndarray, rng: np.random.Generator) -> List[str]:
        added = []
        for gid, v, n in zip(ids, V, nov):
            if n > self.threshold or rng.random() < 0.01:
                self.vecs = np.vstack([self.vecs, v[None]]); self.ids.append(gid); added.append(gid)
        self.recent_adds.append(len(added))
        self.recent_adds = self.recent_adds[-10:]
        rate = sum(self.recent_adds) / max(1, len(ids) * len(self.recent_adds))
        self.threshold *= 1.05 if rate > self.target else 0.95
        self.threshold = float(np.clip(self.threshold, 0.01, 2.0))
        if len(self.ids) > self.max_size:   # mantém os mais isolados (diversidade), remove os mais redundantes
            nv = self.novelty(self.vecs, None)
            keep = np.argsort(-nv)[: self.max_size]
            keep.sort()
            self.vecs = self.vecs[keep]; self.ids = [self.ids[i] for i in keep]
        return added

    def state_dict(self):
        return {"vecs": self.vecs, "ids": self.ids, "threshold": self.threshold, "recent": self.recent_adds}

    def load_state_dict(self, s):
        self.vecs, self.ids, self.threshold, self.recent_adds = np.asarray(s["vecs"], np.float32), list(s["ids"]), s["threshold"], s["recent"]


class Zoo:
    def __init__(self, cap: int = 300):
        self.cap = cap
        self.cats: Dict[str, List[Dict[str, Any]]] = {c: [] for c in ZOO_CATEGORIES}

    @staticmethod
    def classify(d: Dict[str, float], statuses: Dict[str, str], th: Thresholds, anomalous: bool) -> List[str]:
        if d.get("viability") not in ("VIABLE", "FROZEN"):
            return []
        cats = []
        frozen = d["rel_activity"] < th.frozen_rel_activity * 10
        if frozen and d["persistence"] > 0.5 and d["structures"] >= 1:
            cats.append("STATIC")
        if d["periodicity"] > 0.5 and not frozen:
            cats.append("OSCILLATORY")
        if d["causal_spread"] > 0.3 and d["lyapunov"] < th.chaos_lyap:
            cats.append("PROPAGATING")
        if d["persistence"] > th.persistence and not frozen:
            cats.append("PERSISTENT")
        if d["mobility"] > 0.2:
            cats.append("MOBILE")
        if statuses.get("recovery") == STATUS_PASS:
            cats.append("RECOVERING")
        if statuses.get("memory") == STATUS_PASS:
            cats.append("MEMORY_CANDIDATE")
        if statuses.get("channel") == STATUS_PASS:
            cats.append("CHANNEL_CANDIDATE")
        if statuses.get("transformation") == STATUS_PASS:
            cats.append("TRANSFORMER_CANDIDATE")
        if anomalous:
            cats.append("ANOMALOUS")
        if not cats and d.get("viability") == "VIABLE":
            cats.append("UNKNOWN")
        return cats

    def add(self, gid: str, cats: List[str], novelty: float, gen: int):
        for c in cats:
            lst = self.cats[c]
            if any(e["genome_id"] == gid for e in lst):
                continue
            lst.append({"genome_id": gid, "novelty": float(novelty), "generation": gen})
            if len(lst) > self.cap:
                lst.sort(key=lambda e: -e["novelty"])
                del lst[self.cap:]

    def ids(self) -> List[str]:
        return sorted(set(e["genome_id"] for l in self.cats.values() for e in l))

    def counts(self) -> Dict[str, int]:
        return {c: len(v) for c, v in self.cats.items()}

    def state_dict(self):
        return {"cap": self.cap, "cats": self.cats}

    def load_state_dict(self, s):
        self.cap = s["cap"]; self.cats = {c: list(s["cats"].get(c, [])) for c in ZOO_CATEGORIES}


class SimpleArchive:
    def __init__(self, name: str, cap: int = 2000):
        self.name, self.cap = name, cap
        self.entries: List[Dict[str, Any]] = []

    def add(self, gid: str, reason: str, gen: int, **extra) -> bool:
        if any(e["genome_id"] == gid and e["reason"] == reason for e in self.entries):
            return False
        self.entries.append({"genome_id": gid, "reason": reason, "generation": gen, "ts": iso_now(), **to_jsonable(extra)})
        if len(self.entries) > self.cap:
            self.entries = self.entries[-self.cap:]
        return True

    def ids(self) -> List[str]:
        return sorted(set(e["genome_id"] for e in self.entries))

    def state_dict(self):
        return {"name": self.name, "cap": self.cap, "entries": self.entries}

    def load_state_dict(self, s):
        self.name, self.cap, self.entries = s["name"], s["cap"], list(s["entries"])


class Lineage:
    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}
        self.children: Dict[str, List[str]] = {}

    def add(self, g: Genome):
        self.records[g.genome_id] = {"parent_ids": list(g.parent_ids), "generation": g.generation,
                                     "mutation_record": list(g.mutation_record), "birth_timestamp": g.birth_timestamp,
                                     "operator": g.operator, "origin": g.origin, "species_id": g.species_id,
                                     "quality": None, "level": 0}
        for p in g.parent_ids:
            self.children.setdefault(p, []).append(g.genome_id)

    def update(self, gid: str, **kw):
        if gid in self.records:
            self.records[gid].update(to_jsonable(kw))

    def ancestors(self, gid: str, max_depth: int = 200) -> List[str]:
        out, cur, seen = [], gid, set()
        for _ in range(max_depth):
            r = self.records.get(cur)
            if r is None or not r["parent_ids"]:
                break
            cur = r["parent_ids"][0]
            if cur in seen:
                break
            seen.add(cur); out.append(cur)
        return out

    def tree(self, gid: str, depth: int = 3, max_children: int = 6) -> str:
        lines = []

        def rec(node, prefix, d, last):
            r = self.records.get(node, {})
            tag = f"{node} (gen {r.get('generation', '?')}, L{r.get('level', 0)}, {r.get('operator', '')})"
            lines.append(prefix + ("└─ " if last else "├─ ") + tag if prefix or d < depth else tag)
            if d <= 0:
                return
            kids = self.children.get(node, [])[:max_children]
            for i, k in enumerate(kids):
                rec(k, prefix + ("   " if last else "│  "), d - 1, i == len(kids) - 1)
        anc = self.ancestors(gid, 3)
        root = anc[-1] if anc else gid
        lines.append(f"ancestor chain: {' <- '.join([gid] + anc)}")
        rec(root, "", depth, True)
        return "\n".join(lines)

    def prune(self, keep: set, current_gen: int, horizon: int = 30):
        needed = set()
        for gid in keep:
            needed.add(gid); needed.update(self.ancestors(gid))
        for gid in list(self.records.keys()):
            if gid not in needed and current_gen - self.records[gid]["generation"] > horizon:
                del self.records[gid]
        for p in list(self.children.keys()):
            self.children[p] = [c for c in self.children[p] if c in self.records]
            if p not in self.records and not self.children[p]:
                del self.children[p]

    def state_dict(self):
        return {"records": self.records, "children": self.children}

    def load_state_dict(self, s):
        self.records, self.children = dict(s["records"]), {k: list(v) for k, v in s["children"].items()}


class Speciation:
    """Espécies por distância combinada de genótipo e comportamento (fenótipo)."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.reps: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.extinct: List[Dict[str, Any]] = []
        self.births = 0

    def _matrix(self):
        if getattr(self, "_cache", None) is None or self._cache[0] != len(self.reps):
            ids = list(self.reps.keys())
            G = np.stack([self.reps[i]["g"] for i in ids]) if ids else None
            Bm = np.stack([self.reps[i]["b"] for i in ids]) if ids else None
            self._cache = (len(self.reps), ids, G, Bm)
        return self._cache[1], self._cache[2], self._cache[3]

    def assign(self, gvec: np.ndarray, bvec: np.ndarray, gen: int) -> int:
        best, bd = None, 1e9
        ids, G, Bm = self._matrix()
        if ids:
            dg = np.sqrt(np.mean((G - gvec[None]) ** 2, axis=1)) if G.shape[1] == gvec.shape[0] else np.ones(len(ids))
            db = np.sqrt(np.mean((Bm - bvec[None]) ** 2, axis=1))
            d = 0.5 * dg + 0.5 * db
            j = int(np.argmin(d))
            best, bd = ids[j], float(d[j])
        if best is not None and bd < self.threshold:
            self.reps[best]["last_seen"] = gen
            self.reps[best]["members_gen"] = self.reps[best].get("members_gen", 0) + 1
            return best
        sid = self.next_id; self.next_id += 1; self.births += 1
        self.reps[sid] = {"g": gvec.copy(), "b": bvec.copy(), "born": gen, "last_seen": gen, "members_gen": 1}
        return sid

    def end_generation(self, gen: int, alive: set, patience: int = 10):
        for sid in list(self.reps.keys()):
            if sid not in alive and gen - self.reps[sid]["last_seen"] > patience:
                r = self.reps.pop(sid)
                self._cache = None
                self.extinct.append({"species_id": sid, "born": r["born"], "extinct_at": gen})
        for r in self.reps.values():
            r["members_gen"] = 0

    def state_dict(self):
        return {"threshold": self.threshold, "reps": self.reps, "next_id": self.next_id, "extinct": self.extinct[-5000:],
                "births": self.births}

    def load_state_dict(self, s):
        self.threshold, self.next_id, self.births = s["threshold"], s["next_id"], s["births"]
        self.reps = {int(k): v for k, v in s["reps"].items()}
        self.extinct = list(s["extinct"])
        self._cache = None


class SurpriseModel:
    """Modelo histórico do comportamento (média/variância de Welford). Surpresa = RMS dos z-scores."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim); self.m2 = np.zeros(dim)

    def score(self, V: np.ndarray) -> np.ndarray:
        if self.n < 20:
            return np.zeros(len(V))
        sd = np.sqrt(self.m2 / max(1, self.n - 1)) + 0.02
        return np.sqrt(np.mean(((V - self.mean) / sd) ** 2, axis=1))

    def update(self, V: np.ndarray):
        for v in V:
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    def state_dict(self):
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, s):
        self.n, self.mean, self.m2 = s["n"], np.asarray(s["mean"]), np.asarray(s["m2"])


class OperatorBandit:
    """UCB1 sobre operadores de mutação (mede quais produzem melhorias e ajusta probabilidades)."""

    def __init__(self, ops: List[str]):
        self.ops = list(ops)
        self.n = {o: 1.0 for o in ops}
        self.r = {o: 0.5 for o in ops}
        self.total = float(len(ops))

    def select(self, rng: np.random.Generator) -> str:
        if rng.random() < 0.15:
            return str(rng.choice(self.ops))
        ucb = {o: self.r[o] / self.n[o] + math.sqrt(2 * math.log(self.total) / self.n[o]) for o in self.ops}
        return max(ucb, key=ucb.get)

    def update(self, op: str, reward: float):
        if op not in self.n:
            return
        self.n[op] += 1; self.r[op] += float(reward); self.total += 1
        if self.total > 5000:   # esquecimento lento (não-estacionário)
            for o in self.ops:
                self.n[o] *= 0.5; self.r[o] *= 0.5
            self.total = sum(self.n.values())

    def probs(self) -> Dict[str, float]:
        return {o: round(self.r[o] / self.n[o], 3) for o in self.ops}

    def state_dict(self):
        return {"ops": self.ops, "n": self.n, "r": self.r, "total": self.total}

    def load_state_dict(self, s):
        self.ops, self.n, self.r, self.total = s["ops"], s["n"], s["r"], s["total"]
