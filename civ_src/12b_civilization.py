# %% [markdown]
# ## 12b — SETTLEMENT ENGINE · CIVILIZATION ENGINE · HISTORY ENGINE
# Nada é spawnado: assentamentos, cidades, estradas, "casas", trocas e normas são **detectados** a partir do estado.
# * Assentamento = componente espacial persistente com matéria construída **e** agentes (limiares centrais em
#   `Thresholds`). Cidade = assentamento persistente com população, construção, infraestrutura e fluxo acima dos
#   limiares. Linha do tempo por assentamento: fundação detectada, crescimento, declínio, colapso, recuperação.
# * Estrada = trilha de tráfego repetido (tiles com uso acumulado ≥ limiar), nunca decoração. Pontes = aterros usados.
# * "house-like structure" só com evidência funcional: abrigo reduz exposição (física) **e** agentes ocupam abrigos
#   acima da disponibilidade local quando expostos (teste binomial, repetido em janelas).
# * Escada C0–C9 exige combinação multidimensional; população sozinha nunca vira "civilização".
# * `HistoryEngine`: eventos derivados de mudanças reais do estado (nunca texto inventado), com compressão e
#   persistência write-behind.

# %%
def grid_components(mask: np.ndarray, diag: bool = True) -> Tuple[np.ndarray, int]:
    """Componentes conexos de uma grade pequena (BFS em numpy/Python; usada em grades grossas ≤ 128²)."""
    H, W = mask.shape
    lab = np.full((H, W), -1, np.int64)
    n = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if diag else [])
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x] >= 0:
            continue
        stack = [(y, x)]; lab[y, x] = n
        while stack:
            cy, cx = stack.pop()
            for dy, dx in nbrs:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = n; stack.append((yy, xx))
        n += 1
    return lab, n


def gpu_components(mask: torch.Tensor, max_iter: int = 512) -> torch.Tensor:
    """Rótulos de componentes conexos (8-viz.) por propagação de máximo na GPU; 0 = fundo."""
    H, W = mask.shape
    lab = torch.where(mask, torch.arange(1, H * W + 1, device=mask.device, dtype=torch.float32).view(H, W),
                      torch.zeros(H, W, device=mask.device))
    for _ in range(max_iter):
        nl = F.max_pool2d(lab[None, None], 3, 1, 1)[0, 0] * mask
        if torch.equal(nl, lab):
            break
        lab = nl
    return lab.long()


class SettlementEngine:
    def __init__(self, cfg: CivConfig, device: torch.device, cell: int = 8):
        self.cfg, self.device, self.cell = cfg, device, cell
        self.settlements: Dict[int, Dict[str, Any]] = {}
        self.next_sid = 1
        self.road_segments: List[Dict[str, Any]] = []
        self.links: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.shelter = {"k": 0, "n": 0, "p0_sum": 0.0, "windows_sig": 0, "last_p": 1.0}
        self.house_like: Dict[str, Any] = {"labeled": False, "evidence": None}
        self.new_events: List[Dict[str, Any]] = []
        self.last_metrics: Dict[str, Any] = {}

    def _cells(self, ws: WorldState, pool: AgentPool):
        c = self.cell
        Hc, Wc = math.ceil(ws.H / c), math.ceil(ws.W / c)
        built = (ws.struct_h.clamp(min=0) * (ws.struct_integrity < 1e8)).float() + ws.filled.float()
        pad = lambda x: F.pad(x[None, None], (0, Wc * c - ws.W, 0, Hc * c - ws.H))
        blocks = F.avg_pool2d(pad(built), c)[0, 0] * c * c
        encl = F.avg_pool2d(pad((ws.enclosure >= 0.5).float()), c)[0, 0] * c * c
        traffic = F.avg_pool2d(pad(ws.traffic), c)[0, 0]
        alive = pool.alive
        p = pool.pos[alive]
        cy = (p[:, 1] // c).long().clamp(0, Hc - 1); cx = (p[:, 0] // c).long().clamp(0, Wc - 1)
        agents = torch.bincount(cy * Wc + cx, minlength=Hc * Wc).view(Hc, Wc).float()
        return Hc, Wc, blocks, encl, traffic, agents

    def update(self, ws: WorldState, pool: AgentPool, tick: int, social: Optional[SocialEngine],
               rng: np.random.Generator) -> Dict[str, Any]:
        th = self.cfg.thresholds
        Hc, Wc, blocks, encl, traffic, agents = self._cells(ws, pool)
        B, A, E = blocks.cpu().numpy(), agents.cpu().numpy(), encl.cpu().numpy()
        Tr = traffic.cpu().numpy()
        active = ((B >= 1) & (A >= 1)) | (B >= 3)
        lab, n = grid_components(active)
        cands = []
        for i in range(n):
            cells = np.argwhere(lab == i)
            b, a = float(B[lab == i].sum()), float(A[lab == i].sum())
            if b >= th.settlement_min_blocks and a >= th.settlement_min_agents:
                cands.append({"cells": set(map(tuple, cells.tolist())), "blocks": b, "pop": a,
                              "enclosures": float(E[lab == i].sum()), "traffic": float(Tr[lab == i].mean())})
        # estradas: trilhas de tráfego repetido (componentes na GPU)
        road = (ws.traffic >= th.road_min_traffic) & ws.walkable()
        rl = gpu_components(road) if bool(road.any()) else torch.zeros(ws.H, ws.W, dtype=torch.long, device=self.device)
        seg_ids, seg_sizes = torch.unique(rl[rl > 0], return_counts=True) if bool(road.any()) else (torch.zeros(0), torch.zeros(0))
        segs = [(int(s), int(z)) for s, z in zip(seg_ids.tolist(), seg_sizes.tolist()) if z >= 8]
        bridges_used = int((ws.filled & (ws.traffic > 0.1)).sum())
        # casamento com assentamentos anteriores
        matched = set()
        for c in cands:
            best, bo = None, 0
            for sid, s in self.settlements.items():
                if sid in matched or s["state"] == "collapsed" and s.get("gone", 0) > 50:
                    continue
                o = len(c["cells"] & s["cells"])
                if o > bo:
                    best, bo = sid, o
            cy, cx = np.mean([p[0] for p in c["cells"]]), np.mean([p[1] for p in c["cells"]])
            centre = [float((cx + 0.5) * self.cell), float((cy + 0.5) * self.cell)]
            if best is None:
                sid = self.next_sid; self.next_sid += 1
                s = {"id": sid, "state": "candidate", "persistence": 0, "first_seen": tick, "timeline": [],
                     "events": [], "max_pop": 0, "missing": 0}
                self.settlements[sid] = s
            else:
                sid = best; s = self.settlements[sid]
                if s["state"] == "collapsed":
                    s["state"] = "detected"; s["events"].append({"tick": tick, "type": "recovery"})
                    self.new_events.append({"type": "settlement_recovery", "settlement": sid, "tick": tick, "pos": centre})
            matched.add(sid)
            prev_pop = s["timeline"][-1]["pop"] if s["timeline"] else 0
            s.update({"cells": c["cells"], "blocks": c["blocks"], "pop": c["pop"], "enclosures": c["enclosures"],
                      "traffic": c["traffic"], "centre": centre, "persistence": s["persistence"] + 1, "missing": 0,
                      "last_seen": tick, "max_pop": max(s["max_pop"], c["pop"])})
            s["timeline"].append({"tick": tick, "pop": c["pop"], "blocks": c["blocks"], "enclosures": c["enclosures"]})
            s["timeline"] = s["timeline"][-500:]
            if s["state"] == "candidate" and s["persistence"] >= th.settlement_persistence:
                s["state"] = "detected"; s["founded"] = tick
                s["events"].append({"tick": tick, "type": "founded_detected"})
                self.new_events.append({"type": "settlement_founded", "settlement": sid, "tick": tick, "pos": centre,
                                        "pop": c["pop"], "blocks": c["blocks"]})
            elif s["state"] == "detected" and prev_pop > 0:
                if c["pop"] >= 1.5 * prev_pop and c["pop"] - prev_pop >= 4:
                    s["events"].append({"tick": tick, "type": "growth"})
                    self.new_events.append({"type": "settlement_growth", "settlement": sid, "tick": tick, "pos": centre})
            road_inside = sum(1 for (cy_, cx_) in c["cells"] if Tr[cy_, cx_] >= th.road_min_traffic * 0.5)
            if (s["state"] == "detected" and c["pop"] >= th.city_min_population and c["blocks"] >= th.city_min_blocks
                    and road_inside >= 2 and s["persistence"] >= 2 * th.settlement_persistence):
                s["state"] = "city"; s["events"].append({"tick": tick, "type": "city_detected"})
                self.new_events.append({"type": "city_detected", "settlement": sid, "tick": tick, "pos": centre})
        for sid, s in self.settlements.items():
            if sid in matched:
                continue
            s["missing"] = s.get("missing", 0) + 1
            if s["state"] in ("detected", "city") and s["missing"] >= 3:
                s["state"] = "collapsed"; s["events"].append({"tick": tick, "type": "collapse"})
                self.new_events.append({"type": "settlement_collapse", "settlement": sid, "tick": tick,
                                        "pos": s.get("centre", [0, 0])})
            if s["state"] == "candidate" and s["missing"] >= 3:
                s["state"] = "dropped"
            if s["state"] == "collapsed":
                s["gone"] = s.get("gone", 0) + 1
        for sid in [k for k, v in self.settlements.items() if v["state"] == "dropped"]:
            del self.settlements[sid]
        # rede: estradas ligando assentamentos + fluxos de troca entre assentamentos
        live = {sid: s for sid, s in self.settlements.items() if s["state"] in ("detected", "city")}
        self.road_segments = []
        if segs:
            rl_c = rl.cpu().numpy()
            for sgid, size in segs[:200]:
                ys, xs = np.nonzero(rl_c == sgid)
                cells = set(zip((ys // self.cell).tolist(), (xs // self.cell).tolist()))
                touch = [sid for sid, s in live.items() if any((cy + dy, cx + dx) in s["cells"]
                                                                for cy, cx in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1))]
                self.road_segments.append({"id": sgid, "tiles": int(size), "settlements": touch,
                                           "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
                for i in range(len(touch)):
                    for j in range(i + 1, len(touch)):
                        k = (min(touch[i], touch[j]), max(touch[i], touch[j]))
                        L_ = self.links.setdefault(k, {"road": 0, "trade": 0.0, "since": tick})
                        L_["road"] += 1
        if social is not None and len(live) >= 2:
            sid_of = self.assign(pool, live)
            for (l, h), e in list(social.edges.items())[:200000]:
                gw = e[1] + e[2]
                if gw <= 0.02:
                    continue
                sa, sb = sid_of.get(l), sid_of.get(h)
                if sa is not None and sb is not None and sa != sb:
                    k = (min(sa, sb), max(sa, sb))
                    L_ = self.links.setdefault(k, {"road": 0, "trade": 0.0, "since": tick})
                    L_["trade"] += float(gw)
        # abrigo como habitação: ocupância sob exposição × disponibilidade local de abrigos
        self._shelter_evidence(ws, pool, tick)
        n_det = sum(1 for s in self.settlements.values() if s["state"] in ("detected", "city"))
        n_city = sum(1 for s in self.settlements.values() if s["state"] == "city")
        connected = [k for k, v in self.links.items() if (v["road"] > 0 or v["trade"] > 0.1)
                     and k[0] in live and k[1] in live]
        self.last_metrics = {"tick": tick, "settlements": n_det, "cities": n_city,
                             "candidates": sum(1 for s in self.settlements.values() if s["state"] == "candidate"),
                             "road_segments": len(self.road_segments), "road_tiles": int(sum(r["tiles"] for r in self.road_segments)),
                             "bridges_used": bridges_used, "links": len(connected),
                             "blocks_total": float(B.sum()), "enclosures_total": float(E.sum()),
                             "house_like": self.house_like["labeled"], "shelter_p": self.shelter["last_p"]}
        return self.last_metrics

    def assign(self, pool: AgentPool, live: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
        idx = torch.nonzero(pool.alive).flatten()
        aids = pool.aid[idx].cpu().numpy(); p = pool.pos[idx].cpu().numpy()
        cell_owner = {}
        for sid, s in live.items():
            for cc in s["cells"]:
                cell_owner[cc] = sid
        out = {}
        for a, (x, y) in zip(aids.tolist(), p.tolist()):
            sid = cell_owner.get((int(y // self.cell), int(x // self.cell)))
            if sid is not None:
                out[int(a)] = sid
        return out

    def _shelter_evidence(self, ws: WorldState, pool: AgentPool, tick: int):
        th = self.cfg.thresholds
        alive = pool.alive
        if not bool(alive.any()):
            return
        iy, ix = WorldEngine.tile_index(ws, pool.pos[alive])
        expo = (ws.storm[iy, ix] > 0.2) | (ws.temperature[iy, ix] < 0.3)
        sh = ws.enclosure >= 0.5
        if not bool(sh.any()) or not bool(expo.any()):
            return
        c = self.cell
        Hc, Wc = math.ceil(ws.H / c), math.ceil(ws.W / c)
        pad = lambda x: F.pad(x[None, None].float(), (0, Wc * c - ws.W, 0, Hc * c - ws.H))
        frac = (F.avg_pool2d(pad(sh), c) / F.avg_pool2d(pad(ws.walkable() | sh), c).clamp(min=1e-6))[0, 0]
        cyy, cxx = (iy // c).clamp(max=Hc - 1), (ix // c).clamp(max=Wc - 1)
        p0 = frac[cyy, cxx]
        sel = expo & (p0 > 0)
        if not bool(sel.any()):
            return
        k = int((sh[iy, ix] & sel).sum()); n = int(sel.sum()); p0m = float(p0[sel].mean())
        S = self.shelter
        S["k"] += k; S["n"] += n; S["p0_sum"] += p0m * n
        p0_all = S["p0_sum"] / max(1, S["n"])
        p = binom_p_greater(S["k"], S["n"], min(0.999, max(1e-6, p0_all))) if S["n"] >= 30 else 1.0
        S["last_p"] = p
        if p < th.alpha and S["k"] / max(1, S["n"]) > 2 * p0_all:
            S["windows_sig"] += 1
        if S["windows_sig"] >= 2 and not self.house_like["labeled"]:
            self.house_like = {"labeled": True, "evidence": {"tick": tick, "k": S["k"], "n": S["n"], "p0": p0_all, "p": p,
                                                              "damage_reduction": "enclosure multiplies exposure damage by (1 - shelter)"}}
            self.new_events.append({"type": "house_like_structure", "tick": tick, "evidence": self.house_like["evidence"]})

    def drain_events(self) -> List[Dict[str, Any]]:
        ev, self.new_events = self.new_events, []
        return ev

    def public(self, sid: int) -> Optional[Dict[str, Any]]:
        s = self.settlements.get(int(sid))
        if s is None:
            return None
        out = {k: v for k, v in s.items() if k != "cells"}
        out["cells"] = sorted([list(c) for c in s.get("cells", set())])[:400]
        out["cell_size"] = self.cell
        out["links"] = [{"a": a, "b": b, **v} for (a, b), v in self.links.items() if int(sid) in (a, b)]
        return to_jsonable(out)

    def state_dict(self):
        return {"settlements": {str(k): dict(v, cells=sorted([list(c) for c in v.get("cells", set())]))
                                for k, v in self.settlements.items()},
                "next_sid": self.next_sid, "links": [[a, b, v] for (a, b), v in self.links.items()],
                "shelter": self.shelter, "house_like": self.house_like}

    def load_state_dict(self, s):
        self.settlements = {int(k): dict(v, cells=set(tuple(c) for c in v.get("cells", []))) for k, v in s["settlements"].items()}
        self.next_sid = int(s["next_sid"])
        self.links = {(int(a), int(b)): dict(v) for a, b, v in s["links"]}
        self.shelter, self.house_like = dict(s["shelter"]), dict(s["house_like"])


C_LEVELS = ["C0 POPULATION", "C1 STABLE SOCIAL GROUPS", "C2 COMMUNICATION NETWORK", "C3 CULTURAL TRANSMISSION",
            "C4 SETTLEMENT", "C5 SPECIALIZATION", "C6 CUMULATIVE CULTURE", "C7 INFRASTRUCTURE",
            "C8 MULTI-SETTLEMENT NETWORK", "C9 TECHNOLOGICAL CIVILIZATION CANDIDATE"]


class CivilizationEngine:
    """Métricas civilizacionais separadas (nunca um número único) + escada C0–C9 por critérios multidimensionais."""

    def __init__(self, cfg: CivConfig, layout: GenomeLayout):
        self.cfg, self.L = cfg, layout
        self.streak: Dict[str, int] = collections.defaultdict(int)
        self.achieved: Dict[str, Dict[str, Any]] = {}
        self.level = -1
        self.roles_prev: Dict[int, int] = {}
        self.role_centroids: Optional[np.ndarray] = None
        self.last_metrics: Dict[str, Any] = {}
        self.history = RingBuffer(800)
        self.new_events: List[Dict[str, Any]] = []
        self.innov_ticks: "collections.deque" = collections.deque(maxlen=2000)

    def specialization(self, pool: AgentPool, rng: np.random.Generator) -> Dict[str, Any]:
        """Diferenciação de papéis entre adultos: variância entre indivíduos do orçamento de ações acima do nulo
        multinomial (bootstrap paramétrico) + estabilidade dos papéis entre janelas."""
        idx = torch.nonzero(pool.alive & (pool.age > 300)).flatten()
        if idx.numel() < 20:
            return {"spec_p": 1.0, "spec_v": 0.0, "roles": 0, "role_stability": 0.0, "adults": int(idx.numel())}
        sig = pool.signature[idx][:, [6, 7, 8, 10, 11, 12, 2, 3]].clamp(min=0).cpu().numpy() + 1e-3
        frac = sig / sig.sum(1, keepdims=True)
        nt = 50
        counts = np.round(frac * nt)
        pooled = counts.sum(0) / counts.sum()

        def stat(cm):
            f = cm / cm.sum(1, keepdims=True).clip(min=1)
            return float(((f - f.mean(0)) ** 2).sum(1).mean())
        obs = stat(counts)
        null = [stat(rng.multinomial(nt, pooled, size=len(counts)).astype(float)) for _ in range(60)]
        p = (sum(1 for v in null if v >= obs) + 1) / 61
        chi = ((counts - counts.sum(1, keepdims=True) * pooled) ** 2 / (counts.sum(1, keepdims=True) * pooled + 1e-9)).sum()
        V = float(math.sqrt(chi / (counts.sum() * (min(counts.shape) - 1) + 1e-9)))
        k = 4
        C = self.role_centroids if self.role_centroids is not None and self.role_centroids.shape == (k, frac.shape[1]) \
            else frac[rng.choice(len(frac), k, replace=False)]
        for _ in range(8):
            lab = ((frac[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            for j in range(k):
                if (lab == j).any():
                    C[j] = frac[lab == j].mean(0)
        self.role_centroids = C
        aids = pool.aid[idx].cpu().numpy()
        same = [self.roles_prev[a] == r for a, r in zip(aids.tolist(), lab.tolist()) if a in self.roles_prev]
        self.roles_prev = {int(a): int(r) for a, r in zip(aids.tolist(), lab.tolist())}
        return {"spec_p": float(p), "spec_v": V, "roles": int(len(set(lab.tolist()))),
                "role_stability": float(np.mean(same)) if same else 0.0, "adults": int(idx.numel())}

    def evaluate(self, tick: int, pool: AgentPool, agg_total: float, social: Dict[str, Any], comm: Dict[str, Any],
                 culture: Dict[str, Any], settle: Dict[str, Any], validated: Dict[str, bool], rng: np.random.Generator,
                 demography: Dict[str, Any]) -> Dict[str, Any]:
        th = self.cfg.thresholds
        alive = pool.n_alive()
        idx = torch.nonzero(pool.alive).flatten()
        div, species = 0.0, 0
        if idx.numel() >= 10:
            sel = idx[torch.as_tensor(rng.choice(idx.numel(), min(300, idx.numel()), replace=False), device=idx.device)]
            G = pool.genome[sel]
            D = torch.cdist(G / (self.L.hi - self.L.lo).clamp(min=1e-6), G / (self.L.hi - self.L.lo).clamp(min=1e-6)) / math.sqrt(self.L.P)
            div = float(D.mean())
            reps: List[int] = []
            for i in range(D.shape[0]):
                if not reps or float(D[i, reps].min()) > 0.15:
                    reps.append(i)
            species = len(reps)
        spec = self.specialization(pool, rng)
        m = {"tick": tick, "population": alive, "aggregate_population": agg_total, "genetic_diversity": div,
             "species_estimate": species, "births_window": demography.get("births", 0), "deaths_window": demography.get("deaths", 0),
             "culture_types": culture.get("types", 0), "transmission_p": culture.get("transmission_p", 1.0),
             "transmission_coef": culture.get("transmission_coef", 0.0), "chain_depth": culture.get("max_chain_depth", 0),
             "inventions": sum(1 for v in culture.get("techniques", {}).values() if v.get("invented")),
             "blocks": settle.get("blocks_total", 0.0), "enclosures": settle.get("enclosures_total", 0.0),
             "settlements": settle.get("settlements", 0), "cities": settle.get("cities", 0),
             "roads": settle.get("road_segments", 0), "bridges_used": settle.get("bridges_used", 0),
             "links": settle.get("links", 0), "comm_mi": comm.get("mi_world", 0.0), "comm_p": comm.get("p_world", 1.0),
             "comm_resp_p": comm.get("p_response", 1.0), "stable_groups": social.get("stable_groups", 0),
             "modularity_z": social.get("modularity_z", 0.0), "norm_candidate": social.get("norm_candidate", False),
             "reciprocity": social.get("reciprocity", 0.0), **spec,
             "resource_efficiency": demography.get("eaten", 0.0) / max(1e-6, demography.get("agent_ticks", 1.0))}
        self.innov_ticks.extend([tick] * int(demography.get("new_inventions", 0)))
        m["innovation_rate"] = sum(1 for t in self.innov_ticks if tick - t <= 1000)
        crit = {
            "C0 POPULATION": alive + agg_total >= 30 and demography.get("births", 0) > 0,
            "C1 STABLE SOCIAL GROUPS": m["stable_groups"] >= 1 and m["modularity_z"] > 3,
            "C2 COMMUNICATION NETWORK": (m["comm_p"] < th.alpha and m["comm_mi"] > 0.01) or validated.get("H9", False),
            "C3 CULTURAL TRANSMISSION": (m["transmission_p"] < th.alpha and m["transmission_coef"] < 0) or validated.get("H10", False),
            "C4 SETTLEMENT": m["settlements"] >= 1,
            "C5 SPECIALIZATION": m["spec_p"] < th.specialization_p and m["spec_v"] >= 0.15 and m["role_stability"] >= 0.6,
            # profundidade da cadeia sozinha não basta (aquisições "independentes" podem ser genéticas): exige também
            # transmissão social significativa controlando parentesco e ambiente, ou o probe causal H11
            "C6 CUMULATIVE CULTURE": validated.get("H11", False) or (m["chain_depth"] >= 3 and m["transmission_p"] < th.alpha
                                                                      and m["transmission_coef"] < 0),
            "C7 INFRASTRUCTURE": m["roads"] >= 1 or m["bridges_used"] >= 3,
            "C8 MULTI-SETTLEMENT NETWORK": m["settlements"] >= 2 and m["links"] >= 1,
            "C9 TECHNOLOGICAL CIVILIZATION CANDIDATE": False}
        crit["C9 TECHNOLOGICAL CIVILIZATION CANDIDATE"] = (all(crit[c] for c in C_LEVELS[4:9])
                                                           and (validated.get("H7", False) or m["inventions"] >= 4)
                                                           and m["innovation_rate"] > 0)
        need = 3                                                 # janelas consecutivas (persistência)
        for c, ok in crit.items():
            self.streak[c] = self.streak[c] + 1 if ok else 0
            if self.streak[c] >= need and c not in self.achieved:
                self.achieved[c] = {"tick": tick, "evidence": {k: m[k] for k in m if not isinstance(m[k], dict)}}
                self.new_events.append({"type": "civilization_level", "level": c, "tick": tick})
        lvl = -1
        for i, c in enumerate(C_LEVELS):
            if c in self.achieved:
                lvl = i
            else:
                break
        self.level = lvl
        m["level"] = lvl
        m["criteria_now"] = {c: bool(v) for c, v in crit.items()}
        m["achieved"] = sorted(self.achieved.keys())
        self.last_metrics = m
        self.history.push({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
        return m

    def drain_events(self) -> List[Dict[str, Any]]:
        ev, self.new_events = self.new_events, []
        return ev

    def state_dict(self):
        return {"streak": dict(self.streak), "achieved": self.achieved, "level": self.level,
                "roles_prev": list(self.roles_prev.items())[-50000:], "role_centroids": self.role_centroids,
                "history": list(self.history.items), "innov": list(self.innov_ticks)}

    def load_state_dict(self, s):
        self.streak = collections.defaultdict(int, s["streak"]); self.achieved = dict(s["achieved"])
        self.level = int(s["level"]); self.roles_prev = {int(a): int(r) for a, r in s["roles_prev"]}
        self.role_centroids = None if s.get("role_centroids") is None else np.asarray(s["role_centroids"])
        for x in s.get("history", []):
            self.history.push(x)
        self.innov_ticks = collections.deque(s.get("innov", []), maxlen=2000)


class HistoryEngine:
    """Eventos históricos derivados de mudanças reais do estado. Recentes completos; antigos comprimidos em contagens
    por tipo × faixa de 1000 ticks; persistência em lote (write-behind) em history/events.jsonl."""

    def __init__(self, paths: Optional[CivPaths], cap: int = 20000):
        self.paths = paths
        self.events: "collections.deque" = collections.deque(maxlen=cap)
        self.compressed: Dict[str, int] = collections.defaultdict(int)
        self.pending: List[Dict[str, Any]] = []
        self.next_eid = 1
        self.pop_ema: Optional[float] = None
        self.extinct_since: Optional[int] = None
        self.pre_extinction_pop = 0.0
        self.last_species = 0
        self.last_centroid: Optional[np.ndarray] = None
        self.total = 0

    def add(self, etype: str, tick: int, pos: Optional[Sequence[float]] = None, **data) -> Dict[str, Any]:
        if len(self.events) == self.events.maxlen:
            old = self.events[0]
            self.compressed[f"{old['type']}|{old['tick'] // 1000}"] += 1
        ev = {"eid": self.next_eid, "type": etype, "tick": int(tick), "ts": iso_now(), "schema": EVENT_SCHEMA_VERSION,
              "pos": [float(pos[0]), float(pos[1])] if pos is not None else None, "data": to_jsonable(data)}
        self.next_eid += 1
        self.total += 1
        self.events.append(ev)
        self.pending.append(ev)
        return ev

    def observe(self, tick: int, pool: AgentPool, species: int) -> List[Dict[str, Any]]:
        out = []
        n = pool.n_alive()
        if self.pop_ema is None:
            self.pop_ema = float(n)
        if n == 0 and self.extinct_since is None:
            self.extinct_since = tick; self.pre_extinction_pop = self.pop_ema
            out.append(self.add("extinction", tick, population_before=self.pop_ema))
        elif n > 0 and self.extinct_since is not None and n >= max(10, 0.3 * self.pre_extinction_pop):
            out.append(self.add("recovery", tick, population=n, after_ticks=tick - self.extinct_since))
            self.extinct_since = None
        elif self.pop_ema > 20 and abs(n - self.pop_ema) / self.pop_ema > 0.3:
            out.append(self.add("population_shift", tick, population=n, previous=self.pop_ema,
                                change=(n - self.pop_ema) / self.pop_ema))
            self.pop_ema = float(n)
        self.pop_ema = 0.9 * self.pop_ema + 0.1 * n
        if species > self.last_species and self.last_species > 0:
            out.append(self.add("species_split", tick, species=species, previous=self.last_species))
        self.last_species = species
        if n > 10:
            c = pool.pos[pool.alive].mean(0).cpu().numpy()
            if self.last_centroid is not None and np.linalg.norm(c - self.last_centroid) > 6:
                out.append(self.add("migration", tick, pos=c.tolist(), shift=float(np.linalg.norm(c - self.last_centroid))))
            self.last_centroid = c
        return out

    def query(self, since: int = 0, types: Optional[Set[str]] = None, limit: int = 300) -> List[Dict[str, Any]]:
        out = [e for e in self.events if e["tick"] >= since and (types is None or e["type"] in types)]
        return out[-limit:]

    def flush(self):
        if not self.pending or self.paths is None:
            return
        try:
            with open(os.path.join(self.paths.history, "events.jsonl"), "a") as f:
                for e in self.pending:
                    f.write(json.dumps(e) + "\n")
            self.pending = []
        except Exception as e:
            print(f"[history] falha ao gravar: {e}")

    def state_dict(self):
        self.flush()
        return {"events": list(self.events)[-5000:], "compressed": dict(self.compressed), "next_eid": self.next_eid,
                "pop_ema": self.pop_ema, "extinct_since": self.extinct_since, "pre": self.pre_extinction_pop,
                "last_species": self.last_species, "total": self.total}

    def load_state_dict(self, s):
        self.events = collections.deque(s["events"], maxlen=self.events.maxlen)
        self.compressed = collections.defaultdict(int, s["compressed"])
        self.next_eid, self.pop_ema = int(s["next_eid"]), s["pop_ema"]
        self.extinct_since, self.pre_extinction_pop = s["extinct_since"], float(s["pre"])
        self.last_species, self.total = int(s["last_species"]), int(s["total"])
