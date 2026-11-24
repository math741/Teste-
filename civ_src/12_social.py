# %% [markdown]
# ## 11/12 — SOCIAL ENGINE · COMMUNICATION · CULTURE ENGINE · KNOWLEDGE GRAPH
# Tudo aqui é **instrumento externo**: os agentes nunca recebem grafo, rótulo de grupo, "cultura" ou "conhecimento".
# * **SocialEngine** — grafo esparso cujas arestas nascem do histórico real de interação (proximidade, transferência,
#   tomada) com decaimento; comunidades por *label propagation* (detecção externa), grupos estáveis por casamento de
#   Jaccard entre janelas, convivência persistente ("households" sem impor família), reciprocidade e normas funcionais
#   candidatas (retaliação consistente acima da taxa base, teste binomial).
# * **Comunicação** — I(sinal; estado do mundo), I(sinal ouvido; ação seguinte), I(sinal; informação privada) com
#   correção de Miller–Madow e nulo por permutação; sinergia de pares de canais (composicionalidade candidata).
# * **CultureEngine** — tipos comportamentais (k-médias externo nas assinaturas), índice de transmissão social que
#   controla parentesco genético **e** distância espacial (ambiente compartilhado), grafo de conhecimento com inventor,
#   adoções sociais × independentes, mutação cultural e profundidade cumulativa (A cria X → B aprende X e adiciona Y →
#   C aprende X+Y e adiciona Z). Evidência populacional é **candidata**: validação causal fica nos probes H8–H11.

# %%
def mi_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """Informação mútua plug-in (bits) com correção de Miller–Madow para variáveis discretas."""
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    n = a.size
    if n < 4:
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ka, kb = ai.max() + 1, bi.max() + 1
    joint = np.bincount(ai * kb + bi, minlength=ka * kb).reshape(ka, kb).astype(np.float64)
    pj = joint / n
    pa, pb = pj.sum(1), pj.sum(0)
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    bias = ((nz.sum() - (pa > 0).sum() - (pb > 0).sum() + 1) / (2 * n * math.log(2)))
    return max(0.0, mi - bias)


def mi_permutation(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
    obs = mi_discrete(a, b)
    if obs <= 0:
        return 0.0, 1.0
    cnt = 0
    for _ in range(n_perm):
        if mi_discrete(a, rng.permutation(b)) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


class SocialEngine:
    COLS = ["prox", "give_ab", "give_ba", "take_ab", "take_ba"]

    def __init__(self, cfg: CivConfig, tau: float = 1500.0):
        self.cfg, self.tau = cfg, tau
        self.max_edges = max(20000, cfg.max_agents * 8)
        self.edges: Dict[Tuple[int, int], np.ndarray] = {}        # (a<b) → [prox, give_ab, give_ba, take_ab, take_ba, last, n]
        self.adj: Dict[int, Set[int]] = collections.defaultdict(set)
        self.groups: Dict[int, Dict[str, Any]] = {}
        self.next_gid = 1
        self.windows = 0
        self.exposures: "collections.OrderedDict[Tuple[int, int], int]" = collections.OrderedDict()
        self.norm = {"takes": 0, "encounters": 0, "exposed_encounters": 0, "retaliations": 0}
        self.recip = {"give_pairs": 0, "reciprocated": 0}
        self.last_metrics: Dict[str, Any] = {}
        self.history = RingBuffer(600)
        self.comm_history = RingBuffer(600)
        self.prev_sig: Dict[int, int] = {}

    # ------------------------------------------------------------------ ingestão de interações reais
    def ingest(self, rows: Optional[np.ndarray], tick: int):
        if rows is None or len(rows) == 0:
            return
        a, b, k = rows[:, 0], rows[:, 1], rows[:, 2]
        w = rows[:, 3] / 1e4
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        fwd = (a <= b).astype(np.int64)
        key = np.stack([lo, hi, k, fwd], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        wsum = np.bincount(inv, weights=w)
        cnt = np.bincount(inv)
        for (l, h, kk, f), ws_, c in zip(uniq.tolist(), wsum.tolist(), cnt.tolist()):
            e = self.edges.get((l, h))
            if e is None:
                e = np.zeros(7); e[5] = tick
                self.edges[(l, h)] = e
                self.adj[l].add(h); self.adj[h].add(l)
            else:
                e[:5] *= math.exp(-(tick - e[5]) / self.tau); e[5] = tick
            col = 0 if kk == 0 else ((1 if f else 2) if kk == 1 else (3 if f else 4))
            e[col] += ws_; e[6] += c
            if kk == 0:
                self.norm["encounters"] += c
                if (l, h) in self.exposures or (h, l) in self.exposures:
                    self.norm["exposed_encounters"] += c
        # normas funcionais candidatas: retaliação (x tomou de y; depois y toma de x)
        takes = rows[k == 2]
        for x, y, _, _, t in takes.tolist():
            self.norm["takes"] += 1
            if (x, y) in self.exposures and t - self.exposures[(x, y)] <= 500:
                self.norm["retaliations"] += 1
            self.exposures[(y, x)] = t                       # y foi lesado por x → exposição para retaliação de y
            if len(self.exposures) > 50000:
                self.exposures.popitem(last=False)
        if len(self.edges) > self.max_edges:
            self.prune(tick)

    def weight(self, e: np.ndarray, tick: int) -> float:
        return float((e[0] + e[1] + e[2] + 0.5 * (e[3] + e[4])) * math.exp(-(tick - e[5]) / self.tau))

    def prune(self, tick: int):
        items = sorted(self.edges.items(), key=lambda kv: self.weight(kv[1], tick))
        drop = items[: len(items) - int(self.max_edges * 0.8)]
        for (l, h), _ in drop:
            del self.edges[(l, h)]
            self.adj[l].discard(h); self.adj[h].discard(l)
        for k in [k for k, v in self.adj.items() if not v]:
            del self.adj[k]

    def neighbors(self, aid: int, tick: int, k: int = 8) -> List[Tuple[int, float]]:
        out = []
        for o in self.adj.get(int(aid), ()):
            e = self.edges.get((min(aid, o), max(aid, o)))
            if e is not None:
                out.append((int(o), self.weight(e, tick)))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    # ------------------------------------------------------------------ comunidades e grupos estáveis
    def communities(self, alive_aids: np.ndarray, tick: int, min_w: float = 0.02, rng=None) -> Dict[int, int]:
        alive = set(alive_aids.tolist())
        U, V, Wt = [], [], []
        for (l, h), e in self.edges.items():
            if l in alive and h in alive:
                w = self.weight(e, tick)
                if w >= min_w:
                    U.append(l); V.append(h); Wt.append(w)
        if not U:
            return {}
        U, V, Wt = np.asarray(U), np.asarray(V), np.asarray(Wt)
        nodes, inv = np.unique(np.concatenate([U, V]), return_inverse=True)
        n = nodes.size
        ui, vi = inv[: U.size], inv[U.size:]
        labels = np.arange(n)
        src, dst, ww = np.concatenate([ui, vi]), np.concatenate([vi, ui]), np.concatenate([Wt, Wt])
        for _ in range(20):
            key = src * n + labels[dst]
            uk, kinv = np.unique(key, return_inverse=True)
            s = np.bincount(kinv.reshape(-1), weights=ww)
            node, lab = uk // n, uk % n
            own = labels[node] == lab
            s = s + own * 1e-9                                   # empate: mantém o rótulo atual (estabilidade)
            order = np.lexsort((-s, node))
            first = np.ones(order.size, bool); first[1:] = node[order][1:] != node[order][:-1]
            best_node, best_lab = node[order][first], lab[order][first]
            new = labels.copy(); new[best_node] = best_lab
            changed = float((new != labels).mean())
            labels = new
            if changed < 0.005:
                break
        self._last_graph = (ui, vi, Wt, labels)
        return {int(a): int(c) for a, c in zip(nodes.tolist(), labels.tolist())}

    @staticmethod
    def modularity(ui: np.ndarray, vi: np.ndarray, w: np.ndarray, labels: np.ndarray) -> float:
        m = w.sum()
        if m <= 0:
            return 0.0
        n = labels.size
        deg = np.bincount(ui, weights=w, minlength=n) + np.bincount(vi, weights=w, minlength=n)
        same = labels[ui] == labels[vi]
        inside = np.bincount(labels[ui][same], weights=w[same], minlength=n)
        dsum = np.bincount(labels, weights=deg, minlength=n)
        return float((inside / m - (dsum / (2 * m)) ** 2).sum())

    def update_groups(self, pool: AgentPool, tick: int, rng: np.random.Generator) -> Dict[str, Any]:
        th = self.cfg.thresholds
        alive_idx = torch.nonzero(pool.alive).flatten()
        aids = pool.aid[alive_idx].cpu().numpy()
        pos = pool.pos[alive_idx].cpu().numpy()
        pos_of = {int(a): p for a, p in zip(aids.tolist(), pos)}
        comm = self.communities(aids, tick)
        self.windows += 1
        members: Dict[int, Set[int]] = collections.defaultdict(set)
        for a, c in comm.items():
            members[c].add(a)
        cands = [m for m in members.values() if len(m) >= th.group_min_size]
        # casamento com grupos anteriores (Jaccard ≥ 0.3, guloso por sobreposição)
        matched, new_groups = set(), {}
        for m in sorted(cands, key=len, reverse=True):
            best, bj = None, 0.3
            for gid, g in self.groups.items():
                if gid in matched:
                    continue
                inter = len(m & g["members"])
                if inter == 0:
                    continue
                j = inter / len(m | g["members"])
                if j >= bj:
                    best, bj = gid, j
            if best is None:
                gid = self.next_gid; self.next_gid += 1
                g = {"born": tick, "age": 0}
            else:
                gid = best; g = self.groups[best]; matched.add(best)
            P = np.stack([pos_of[a] for a in m if a in pos_of]) if m else np.zeros((0, 2))
            cen = P.mean(0) if len(P) else np.zeros(2)
            g.update({"members": m, "age": g["age"] + 1, "size": len(m), "last": tick, "centroid": cen.tolist(),
                      "spread": float(np.sqrt(((P - cen) ** 2).sum(1)).mean()) if len(P) else 0.0})
            new_groups[gid] = g
        dissolved = [gid for gid in self.groups if gid not in new_groups and self.groups[gid]["age"] >= th.group_persistence]
        self.groups = new_groups
        stable = [gid for gid, g in self.groups.items() if g["age"] >= th.group_persistence]
        households = [gid for gid in stable if self.groups[gid]["spread"] <= 4.0]
        Q, z = 0.0, 0.0
        if comm and hasattr(self, "_last_graph"):
            ui, vi, w, labels = self._last_graph
            Q = self.modularity(ui, vi, w, labels)
            null = [self.modularity(ui, rng.permutation(vi), w, labels) for _ in range(8)]
            z = (Q - float(np.mean(null))) / (float(np.std(null)) + 1e-9)
        n = self.norm
        base = n["takes"] / max(1, n["encounters"])
        ret_rate = n["retaliations"] / max(1, n["exposed_encounters"])
        p_norm = binom_p_greater(n["retaliations"], max(1, n["exposed_encounters"]), min(0.999, max(base, 1e-6))) \
            if n["exposed_encounters"] > 20 else 1.0
        gives = [(e[1] > 0.01, e[2] > 0.01) for e in self.edges.values()]
        g_any = sum(1 for a, b in gives if a or b)
        g_both = sum(1 for a, b in gives if a and b)
        pa = sum(1 for a, b in gives if a) / max(1, g_any); pb = sum(1 for a, b in gives if b) / max(1, g_any)
        recip = g_both / max(1, g_any)
        recip_null = pa * pb
        self.last_metrics = {"tick": tick, "edges": len(self.edges), "communities": len(members),
                             "groups": len(self.groups), "stable_groups": len(stable), "households": len(households),
                             "modularity": Q, "modularity_z": z, "dissolved": dissolved,
                             "retaliation_rate": ret_rate, "take_base_rate": base, "norm_p": p_norm,
                             "norm_candidate": bool(p_norm < th.alpha and ret_rate > 2 * base),
                             "reciprocity": recip, "reciprocity_null": recip_null,
                             "largest_group": max([g["size"] for g in self.groups.values()], default=0)}
        self.history.push({k: v for k, v in self.last_metrics.items() if k != "dissolved"})
        return self.last_metrics

    # ------------------------------------------------------------------ comunicação (fundamentação de sinais)
    def communication(self, pool: AgentPool, ws: WorldState, rng: np.random.Generator, n_perm: int, tick: int,
                      max_n: int = 3000) -> Dict[str, Any]:
        idx = torch.nonzero(pool.alive).flatten()
        if idx.numel() < 30:
            return {}
        if idx.numel() > max_n:
            idx = idx[torch.as_tensor(rng.choice(idx.numel(), max_n, replace=False), device=idx.device)]
        emit = pool.emit[idx].cpu().numpy()
        S = pool.sense_cache[idx].cpu().numpy()
        out = pool.out[idx].cpu().numpy()
        aid = pool.aid[idx].cpu().numpy()
        ns = emit.shape[1]
        loud = emit.max(1) > 0.3
        sig = np.where(loud, emit.argmax(1), ns)
        resb = np.digitize(S[:, 1], [0.1, 0.4])
        state = resb * 8 + (S[:, 29] > 0.2) * 4 + (S[:, 4] > 0.5) * 2 + (S[:, 10] > 0.3)
        heard = S[:, 12:16]
        hcls = np.where(heard.max(1) > 0.2, heard.argmax(1), 4)
        act = 1 / (1 + np.exp(-out))
        acls = (act[:, 6] > 0.5) * 4 + (np.tanh(out[:, 0]) > 0.3) * 2 + (act[:, 2:6].max(1) > 0.5)
        hidden = np.sign(np.round(S[:, 2] * 4)).astype(np.int64) + 1        # gradiente de recurso privado (local)
        n_perm = max(10, n_perm // 2)
        r = {"tick": tick, "n": int(len(sig)), "loud_fraction": float(loud.mean())}
        if loud.mean() < 0.01 or loud.mean() > 0.99:
            r.update({"mi_world": 0.0, "p_world": 1.0, "mi_response": 0.0, "p_response": 1.0, "mi_hidden": 0.0,
                      "p_hidden": 1.0, "synergy": 0.0, "p_synergy": 1.0})
        else:
            r["mi_world"], r["p_world"] = mi_permutation(sig, state, n_perm, rng)
            r["mi_response"], r["p_response"] = mi_permutation(hcls, acls, n_perm, rng)
            r["mi_hidden"], r["p_hidden"] = mi_permutation(sig, hidden, n_perm, rng)
            # composicionalidade candidata: o PAR dos dois canais mais altos informa além do melhor canal isolado
            top2 = np.argsort(-emit, 1)[:, :2]
            pair = np.where(loud, np.minimum(top2[:, 0], top2[:, 1]) * ns + np.maximum(top2[:, 0], top2[:, 1]), ns * ns)
            single = max(mi_discrete(np.where(loud, top2[:, 0], ns), state), mi_discrete(np.where(loud, top2[:, 1], ns), state))
            syn = mi_discrete(pair, state) - single
            null = [mi_discrete(rng.permutation(pair), state) - single for _ in range(n_perm // 2)]
            r["synergy"] = float(syn)
            r["p_synergy"] = float((sum(1 for x in null if x >= syn) + 1) / (len(null) + 1)) if syn > 0 else 1.0
        # I(sinal; ação futura) — sinal na janela anterior × ação atual do mesmo indivíduo-alvo (vizinho mais próximo)
        cur = {int(a): int(c) for a, c in zip(aid.tolist(), acls.tolist())}
        partners = pool.last_partner[idx].cpu().numpy()
        pairs = [(self.prev_sig[int(p)], cur[int(a)]) for a, p in zip(aid.tolist(), partners.tolist())
                 if int(p) in self.prev_sig and int(a) in cur]
        if len(pairs) > 40:
            pa_, pb_ = np.asarray(pairs).T
            r["mi_future"], r["p_future"] = mi_permutation(pa_, pb_, n_perm, rng)
        else:
            r["mi_future"], r["p_future"] = 0.0, 1.0
        self.prev_sig = {int(a): int(s) for a, s in zip(aid.tolist(), sig.tolist())}
        self.comm_history.push(r)
        return r

    def state_dict(self):
        keys = list(self.edges.keys())
        return {"keys": np.asarray(keys, np.int64).reshape(-1, 2), "vals": np.stack([self.edges[k] for k in keys]) if keys
                else np.zeros((0, 7)), "groups": {str(g): dict(v, members=sorted(v["members"])) for g, v in self.groups.items()},
                "next_gid": self.next_gid, "windows": self.windows, "norm": self.norm,
                "history": list(self.history.items), "comm_history": list(self.comm_history.items)}

    def load_state_dict(self, s):
        self.edges, self.adj = {}, collections.defaultdict(set)
        for (l, h), v in zip(np.asarray(s["keys"]).tolist(), np.asarray(s["vals"])):
            self.edges[(int(l), int(h))] = np.asarray(v, np.float64)
            self.adj[int(l)].add(int(h)); self.adj[int(h)].add(int(l))
        self.groups = {int(g): dict(v, members=set(v["members"])) for g, v in s["groups"].items()}
        self.next_gid, self.windows, self.norm = int(s["next_gid"]), int(s["windows"]), dict(s["norm"])
        for x in s.get("history", []):
            self.history.push(x)
        for x in s.get("comm_history", []):
            self.comm_history.push(x)


class CultureEngine:
    """Cultura operacional: informação adquirida socialmente que persiste entre indivíduos (medida de fora)."""

    def __init__(self, cfg: CivConfig, layout: GenomeLayout, k_types: int = 6):
        self.cfg, self.L, self.k = cfg, layout, k_types
        self.centroids: Optional[np.ndarray] = None
        self.inventions: Dict[int, Dict[str, Any]] = {}           # bit → inventor, tick, pos
        self.practitioners: Dict[int, "collections.OrderedDict[int, int]"] = {b: collections.OrderedDict() for b in range(N_TECH)}
        self.adoptions = {b: {"social": 0, "independent": 0} for b in range(N_TECH)}
        self.transmissions: "collections.deque" = collections.deque(maxlen=20000)
        self.mutations: "collections.deque" = collections.deque(maxlen=5000)
        self.depth: "collections.OrderedDict[int, int]" = collections.OrderedDict()
        self.social_bits: "collections.OrderedDict[int, int]" = collections.OrderedDict()
        self.max_depth = 0
        self.chain_examples: "collections.deque" = collections.deque(maxlen=50)
        self.last_metrics: Dict[str, Any] = {}
        self.history = RingBuffer(600)
        self.new_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ eventos de técnicas (do AgentEngine)
    def ingest_tech(self, events: "collections.deque", social: SocialEngine, tick: int):
        while events:
            aid_t, bits_t, partner_t, t, pos_t = events.popleft()
            aids, bits, partners, pos = aid_t.cpu().numpy(), bits_t.cpu().numpy(), partner_t.cpu().numpy(), pos_t.cpu().numpy()
            for a, bb, p, xy in zip(aids.tolist(), bits.tolist(), partners.tolist(), pos):
                for b in range(N_TECH):
                    if not (bb >> b) & 1:
                        continue
                    pr = self.practitioners[b]
                    nb = social.adj.get(a, set())
                    src = next((o for o in ([p] if p in pr else []) + [o for o in nb if o in pr]), None)
                    if b not in self.inventions:
                        self.inventions[b] = {"tech": TECH_NAMES[b], "inventor": a, "tick": t, "pos": xy.tolist()}
                        self.new_events.append({"type": "invention", "tech": TECH_NAMES[b], "agent": a, "tick": t,
                                                "pos": xy.tolist()})
                        self.depth[a] = max(self.depth.get(a, 0), 1)
                    elif src is not None:
                        self.adoptions[b]["social"] += 1
                        self.transmissions.append((src, a, b, t))
                        self.depth[a] = max(self.depth.get(a, 0), self.depth.get(src, 0))
                        self.social_bits[a] = self.social_bits.get(a, 0) | (1 << b)
                    else:
                        self.adoptions[b]["independent"] += 1
                        if self.social_bits.get(a, 0):
                            # acrescentou técnica nova para a vizinhança depois de aprender socialmente → passo cumulativo
                            d = self.depth.get(a, 1) + 1
                            self.depth[a] = d
                            self.mutations.append((a, b, t))
                            if d > self.max_depth:
                                self.max_depth = d
                                self.chain_examples.append({"agent": a, "depth": d, "tech": TECH_NAMES[b], "tick": t})
                                if d >= 3:
                                    self.new_events.append({"type": "cumulative_chain", "depth": d, "agent": a,
                                                            "tech": TECH_NAMES[b], "tick": t, "pos": xy.tolist()})
                    pr[a] = t
                    if len(pr) > 5000:
                        pr.popitem(last=False)
            for od in (self.depth, self.social_bits):
                while len(od) > 200000:
                    od.popitem(last=False)

    # ------------------------------------------------------------------ tipos culturais e transmissão social
    def update(self, pool: AgentPool, social: SocialEngine, tick: int, rng: np.random.Generator) -> Dict[str, Any]:
        idx = torch.nonzero(pool.alive).flatten()
        n = int(idx.numel())
        m: Dict[str, Any] = {"tick": tick, "n": n}
        if n >= 20:
            sig = pool.signature[idx]
            k = int(min(self.k, max(2, n // 20)))
            samp = sig[torch.as_tensor(rng.choice(n, min(n, 4000), replace=False), device=sig.device)].cpu().numpy()
            C = self.centroids if self.centroids is not None and self.centroids.shape == (k, samp.shape[1]) \
                else samp[rng.choice(len(samp), k, replace=False)]
            for _ in range(10):
                lab = ((samp[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
                for j in range(k):
                    if (lab == j).any():
                        C[j] = samp[lab == j].mean(0)
            self.centroids = C
            Ct = torch.as_tensor(C, device=sig.device)
            pool.culture_type[idx] = torch.cdist(sig, Ct).argmin(1).int()
            counts = torch.bincount(pool.culture_type[idx].long(), minlength=k).cpu().numpy()
            pr = counts / max(1, counts.sum())
            m["types"] = int((counts > 0).sum())
            m["type_entropy"] = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
            m.update(self.transmission_index(pool, idx, social, tick, rng))
        m["techniques"] = {TECH_NAMES[b]: {"invented": b in self.inventions, **self.adoptions[b],
                                          "practitioners": len(self.practitioners[b])} for b in range(N_TECH)}
        m["max_chain_depth"] = self.max_depth
        m["transmissions"] = len(self.transmissions)
        m["mutations"] = len(self.mutations)
        soc = sum(v["social"] for v in self.adoptions.values()); ind = sum(v["independent"] for v in self.adoptions.values())
        m["social_adoption_fraction"] = soc / max(1, soc + ind)
        self.last_metrics = m
        self.history.push({k2: v for k2, v in m.items() if k2 != "techniques"})
        return m

    def transmission_index(self, pool, idx, social: SocialEngine, tick: int, rng) -> Dict[str, Any]:
        """Distância comportamental ~ genética + espacial + laço social. Coeficiente do laço < 0 com p (permutação)
        indica similaridade social além de parentesco e ambiente compartilhado (evidência candidata, não causal)."""
        aids = pool.aid[idx].cpu().numpy()
        slot = {int(a): i for i, a in enumerate(aids.tolist())}
        linked = []
        for (l, h), e in social.edges.items():
            if l in slot and h in slot and social.weight(e, tick) > 0.05:
                linked.append((slot[l], slot[h]))
            if len(linked) >= 1500:
                break
        if len(linked) < 30:
            return {"transmission_coef": 0.0, "transmission_p": 1.0, "linked_pairs": len(linked)}
        nl = len(linked)
        rnd = np.stack([rng.integers(0, len(aids), nl), rng.integers(0, len(aids), nl)], 1)
        P = np.concatenate([np.asarray(linked), rnd])
        tie = np.concatenate([np.ones(nl), np.zeros(nl)])
        I0 = torch.as_tensor(idx[torch.as_tensor(P[:, 0], device=idx.device)]); I1 = idx[torch.as_tensor(P[:, 1], device=idx.device)]
        bd = (pool.signature[I0] - pool.signature[I1]).norm(dim=-1).cpu().numpy()
        gd = self.L.distance(pool.genome[I0], pool.genome[I1]).cpu().numpy()
        sd = (pool.pos[I0] - pool.pos[I1]).norm(dim=-1).cpu().numpy()
        X = np.stack([np.ones_like(gd), gd, np.log1p(sd), tie], 1)

        def coef(t):
            X[:, 3] = t
            beta, *_ = np.linalg.lstsq(X, bd, rcond=None)
            return beta[3]
        obs = coef(tie)
        null = [coef(rng.permutation(tie)) for _ in range(40)]
        p = (sum(1 for v in null if v <= obs) + 1) / (len(null) + 1)
        return {"transmission_coef": float(obs), "transmission_p": float(p), "linked_pairs": nl}

    def drain_events(self) -> List[Dict[str, Any]]:
        ev, self.new_events = self.new_events, []
        return ev

    def state_dict(self):
        return {"centroids": self.centroids, "inventions": {str(k): v for k, v in self.inventions.items()},
                "practitioners": {str(b): list(od.items())[-2000:] for b, od in self.practitioners.items()},
                "adoptions": {str(k): v for k, v in self.adoptions.items()}, "transmissions": list(self.transmissions)[-5000:],
                "mutations": list(self.mutations)[-2000:], "depth": list(self.depth.items())[-50000:],
                "social_bits": list(self.social_bits.items())[-50000:], "max_depth": self.max_depth,
                "chain_examples": list(self.chain_examples), "history": list(self.history.items)}

    def load_state_dict(self, s):
        self.centroids = None if s.get("centroids") is None else np.asarray(s["centroids"], np.float32)
        self.inventions = {int(k): v for k, v in s["inventions"].items()}
        for b, items in s["practitioners"].items():
            self.practitioners[int(b)] = collections.OrderedDict((int(a), int(t)) for a, t in items)
        self.adoptions = {int(k): dict(v) for k, v in s["adoptions"].items()}
        self.transmissions = collections.deque([tuple(x) for x in s["transmissions"]], maxlen=20000)
        self.mutations = collections.deque([tuple(x) for x in s["mutations"]], maxlen=5000)
        self.depth = collections.OrderedDict((int(a), int(d)) for a, d in s["depth"])
        self.social_bits = collections.OrderedDict((int(a), int(d)) for a, d in s["social_bits"])
        self.max_depth = int(s["max_depth"]); self.chain_examples = collections.deque(s["chain_examples"], maxlen=50)
        for x in s.get("history", []):
            self.history.push(x)
