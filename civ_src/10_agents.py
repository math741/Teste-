# %% [markdown]
# ## 10 — AGENT ENGINE: indivíduos multiescala (F0 → F4)
# Ciclo por tick (tudo vetorizado; vizinhança por hash espacial na GPU, nunca O(N²)):
# **sentir** (campos físicos do tile, gradientes no referencial do agente, vizinho mais próximo, interocepção) →
# **sinal** (módulo 09) → **agir** (mover, emitir som, exibir cor, consumir, pegar/depositar matéria, marcar, transferir,
# tomar, reproduzir, repousar) → **metabolismo** (toda ação custa recurso; exposição a tempestade/frio danifica,
# abrigo reduz; fome destrói integridade; reparo custa) → morte funcional (integridade ≤ 0 ou idade > limite do genoma).
#
# Escada de fidelidade: **F0** agregado estatístico por região (demografia logística + migração + reservatório de
# genomas) · **F1** agente leve (marcha do corpo resumida por medição física real do genoma) · **F2** corpo
# simplificado (grade de desenvolvimento D−1) · **F3** corpo completo (Matter Nodes) · **F4** foco científico (corpo
# completo + gravação). Promoção/rebaixamento por novidade, ancestralidade, raridade, pedido do Scientist ou do
# administrador (registrado como intervenção). A câmera do usuário **nunca** muda a fidelidade científica.

# %%
@dataclass
class ArenaControl:
    no_repro: bool = False
    no_death: bool = False
    resource_visible: Optional[torch.Tensor] = None   # [hi] bool
    mute: Optional[torch.Tensor] = None               # [hi] bool (emissão sonora bloqueada)
    freeze_move: Optional[torch.Tensor] = None        # [hi] bool
    damage: Optional[torch.Tensor] = None             # [hi] dano extra neste tick
    iv: Optional[MindIntervention] = None
    sound_override: Optional[Callable] = None         # f(sound_in [hi,4]) -> sound_in (troca/embaralha canal)
    energy_floor: Optional[float] = None
    action_block: Optional[torch.Tensor] = None       # [hi, N_ACTIONS] bool (ablação de canais de ação)
    action_force: Optional[torch.Tensor] = None       # [hi, N_ACTIONS] bool (porta forçada aberta — só em arenas)


def spatial_neighbors(pos: torch.Tensor, alive: torch.Tensor, R: float, M: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hash espacial uniforme: vizinho mais próximo, distância e contagem dentro do raio R (até M candidatos por célula)."""
    n = pos.shape[0]
    d = pos.device
    if n == 0:
        z = torch.zeros(0, device=d)
        return z.long(), z, z
    cx = (pos[:, 0] / R).floor().long() + 1
    cy = (pos[:, 1] / R).floor().long() + 1
    ncx = int(cx.max().item()) + 3
    BIG = 1 << 60
    key = torch.where(alive, cy * ncx + cx, torch.full_like(cx, BIG))
    order = torch.argsort(key)
    skey = key[order]
    cands, valid = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k2 = (cy + dy) * ncx + (cx + dx)
            st = torch.searchsorted(skey, k2)
            en = torch.searchsorted(skey, k2, right=True)
            for j in range(M):
                ii = st + j
                cands.append(order[ii.clamp(max=n - 1)])
                valid.append(ii < en)
    C = torch.stack(cands, 1)
    V = torch.stack(valid, 1) & alive.unsqueeze(1) & (C != torch.arange(n, device=d).unsqueeze(1))
    dist = (pos[C] - pos.unsqueeze(1)).norm(dim=-1)
    V = V & (dist <= R)
    dist = torch.where(V, dist, torch.full_like(dist, 1e9))
    best = dist.argmin(1)
    nn = torch.gather(C, 1, best.unsqueeze(1)).squeeze(1)
    nd = torch.gather(dist, 1, best.unsqueeze(1)).squeeze(1)
    has = V.any(1)
    return torch.where(has, nn, torch.full_like(nn, -1)), torch.where(has, nd, torch.full_like(nd, 1e9)), V.sum(1).float()


class BodyRegistry:
    """Corpos ativos (F2+) numa grade fixa de capacidade; slots inativos não têm nós ativos (custo previsível)."""

    def __init__(self, cfg: CivConfig, dev_engine: DevelopmentEngine, body_engine: BodyEngine, cap: int, substeps: int,
                 record: bool = False):
        self.cfg, self.dev, self.body, self.cap, self.substeps, self.record = cfg, dev_engine, body_engine, int(cap), substeps, record
        d = dev_engine.device
        B, N = self.cap, dev_engine.N
        empty = Blueprint(occ=torch.zeros(B, N, dtype=torch.bool, device=d), props=torch.zeros(B, N, N_PROPS, device=d),
                          morph=torch.zeros(B, cfg.n_morph, N, device=d), innerv=torch.zeros(B, N, N_MOTOR, device=d),
                          phase=torch.zeros(B, N, device=d), osc=torch.ones(B, device=d),
                          parent_cell=torch.full((B, N), -1, dtype=torch.long, device=d),
                          birth_step=torch.full((B, N), -1, dtype=torch.long, device=d))
        self.bb = body_engine.make(empty)
        self.agent_slot = torch.full((B,), -1, dtype=torch.long, device=d)
        self.agent_aid = torch.zeros(B, dtype=torch.int64, device=d)
        self.dev_frames: Dict[int, List[np.ndarray]] = {}
        self.trace: Dict[int, "collections.deque"] = {}

    def free(self) -> torch.Tensor:
        return torch.nonzero(self.agent_slot < 0).flatten()

    def attach(self, agent_slots: torch.Tensor, aids: torch.Tensor, genomes: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        free = self.free()[: agent_slots.numel()]
        n = free.numel()
        if n == 0:
            return free
        bp = self.dev.develop(genomes[:n], seeds[:n], record=self.record)
        nb = self.body.make(bp)
        for f in BodyBatch.FIELDS:
            getattr(self.bb, f)[free] = getattr(nb, f)
        if hasattr(self.bb, "_ang_prev"):
            P = self.bb.pos[free]
            self.bb._ang_prev[free] = torch.atan2(P[..., 1], P[..., 0])
        self.agent_slot[free] = agent_slots[:n]
        self.agent_aid[free] = aids[:n]
        if self.record and bp.frames is not None:
            for j, bs in enumerate(free.tolist()):
                self.dev_frames[bs] = [f_[j].cpu().numpy() for f_ in bp.frames]
                self.trace[bs] = collections.deque(maxlen=400)
        return free

    def detach(self, body_slots: torch.Tensor):
        if body_slots.numel() == 0:
            return
        self.bb.active[body_slots] = False
        self.bb.e_on[body_slots] = False
        self.agent_slot[body_slots] = -1
        for bs in body_slots.tolist():
            self.dev_frames.pop(bs, None); self.trace.pop(bs, None)

    def step(self, motor: torch.Tensor) -> Dict[str, torch.Tensor]:
        r = self.body.step(self.bb, motor, substeps=self.substeps)
        if self.record:
            act = torch.nonzero(self.agent_slot >= 0).flatten().tolist()
            if act:
                P = self.bb.pos[act].detach().cpu().numpy().astype(np.float16)
                Sg = self.bb.signal[act].detach().cpu().numpy().astype(np.float16)
                for j, bs in enumerate(act):
                    if bs in self.trace:
                        self.trace[bs].append((P[j], Sg[j]))
        bad = ~r["finite"]
        if bool(bad.any()):
            self.detach(torch.nonzero(bad & (self.agent_slot >= 0)).flatten())
        return r


class RegionAggregates:
    """F0: populações distantes como estado demográfico (contagem, energia média, assinatura cultural média,
    reservatório de genomas reais para desagregação consistente)."""

    def __init__(self, cfg: CivConfig, H: int, W: int, P: int):
        self.cfg, self.rs = cfg, cfg.region_size
        self.Rh, self.Rw = math.ceil(H / self.rs), math.ceil(W / self.rs)
        self.count = np.zeros((self.Rh, self.Rw), np.float64)
        self.energy = np.zeros((self.Rh, self.Rw), np.float64)
        self.signature = np.zeros((self.Rh, self.Rw, N_ACTIONS), np.float32)
        self.reservoir: Dict[int, List[np.ndarray]] = {}
        self.seen: Dict[int, int] = {}
        self.births = 0.0
        self.deaths = 0.0
        self.P = P

    def region_of(self, pos: np.ndarray) -> np.ndarray:
        ry = np.clip((pos[:, 1] // self.rs).astype(int), 0, self.Rh - 1)
        rx = np.clip((pos[:, 0] // self.rs).astype(int), 0, self.Rw - 1)
        return ry * self.Rw + rx

    def absorb(self, regions: np.ndarray, genomes: np.ndarray, energy: np.ndarray, sig: np.ndarray, rng: np.random.Generator):
        for r, g, e, s in zip(regions.tolist(), genomes, energy.tolist(), sig):
            ry, rx = divmod(r, self.Rw)
            c = self.count[ry, rx]
            self.energy[ry, rx] = (self.energy[ry, rx] * c + e) / (c + 1)
            self.signature[ry, rx] = (self.signature[ry, rx] * c + s) / (c + 1)
            self.count[ry, rx] = c + 1
            res = self.reservoir.setdefault(r, [])
            self.seen[r] = self.seen.get(r, 0) + 1
            if len(res) < 8:
                res.append(g.astype(np.float32))
            else:
                j = int(rng.integers(0, self.seen[r]))
                if j < 8:
                    res[j] = g.astype(np.float32)

    def update(self, ws: WorldState, dt: float, rng: np.random.Generator):
        """Demografia logística com capacidade dada pelo recurso real da região; consome recurso do mundo; migração."""
        if self.count.sum() <= 0:
            return
        res = F.adaptive_avg_pool2d(ws.resource[None, None], (self.Rh, self.Rw))[0, 0].cpu().numpy() * self.rs * self.rs
        cap = res * 1.2
        r = 0.002 * dt
        growth = r * self.count * (1 - self.count / np.maximum(cap, 1e-6))
        births = np.clip(growth, 0, None); deaths = np.clip(-growth, 0, None)
        self.births += float(births.sum()); self.deaths += float(deaths.sum())
        self.count = np.clip(self.count + growth, 0, None)
        eat = np.minimum(res, self.count * self.cfg.basal_cost * dt)
        frac = np.where(res > 0, eat / np.maximum(res, 1e-9), 0)
        mult = torch.as_tensor(1 - frac, device=ws.device, dtype=torch.float32)
        up = mult.repeat_interleave(self.rs, 0).repeat_interleave(self.rs, 1)[: ws.H, : ws.W]
        ws.resource = ws.resource * up
        mig = 0.01 * dt * self.count
        flow = np.zeros_like(self.count)
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb_cap = np.roll(np.roll(cap, dy, 0), dx, 1)
            nb_cnt = np.roll(np.roll(self.count, dy, 0), dx, 1)
            push = mig / 4 * (nb_cap > nb_cnt)
            flow -= push
            flow += np.roll(np.roll(push, -dy, 0), -dx, 1)
        self.count = np.clip(self.count + flow, 0, None)

    def total(self) -> float:
        return float(self.count.sum())

    def state_dict(self):
        return {"count": self.count, "energy": self.energy, "signature": self.signature,
                "reservoir": {int(k): np.stack(v) for k, v in self.reservoir.items() if v}, "seen": self.seen,
                "births": self.births, "deaths": self.deaths}

    def load_state_dict(self, s):
        self.count, self.energy, self.signature = np.asarray(s["count"]), np.asarray(s["energy"]), np.asarray(s["signature"])
        self.reservoir = {int(k): [x for x in np.asarray(v)] for k, v in s["reservoir"].items()}
        self.seen = {int(k): int(v) for k, v in s["seen"].items()}
        self.births, self.deaths = float(s["births"]), float(s["deaths"])


class InteractionBuffer:
    """Interações (proximidade, transferência, tomada) acumuladas na GPU e drenadas pelo motor social."""

    def __init__(self, cap_ticks: int = 64):
        self.items: "collections.deque" = collections.deque(maxlen=cap_ticks)

    def push(self, a: torch.Tensor, b: torch.Tensor, kind: int, w: torch.Tensor, tick: int):
        if a.numel():
            self.items.append((a, b, torch.full_like(a, kind), w, tick))

    def drain(self) -> Optional[np.ndarray]:
        if not self.items:
            return None
        A = torch.cat([x[0] for x in self.items]); B = torch.cat([x[1] for x in self.items])
        K = torch.cat([x[2] for x in self.items]); Wt = torch.cat([x[3] for x in self.items])
        T = torch.cat([torch.full_like(x[0], x[4]) for x in self.items])
        self.items.clear()
        return torch.stack([A, B, K, (Wt * 1e4).long(), T], 1).cpu().numpy()


class AgentEngine:
    BASE_TURN = 0.25                      # giro basal (rad/tick) comum a todos os níveis de fidelidade
    MAX_BODY_TURN = 0.6                   # limite do termo de giro vindo do corpo (rad/tick)
    MAX_WOBBLE = 0.4                      # limite do termo difusivo (tiles/tick)

    def __init__(self, cfg: CivConfig, layout: GenomeLayout, world_engine: WorldEngine, matter: MatterEngine,
                 mind: MindEngine, device: torch.device):
        self.cfg, self.L, self.we, self.matter, self.mind, self.device = cfg, layout, world_engine, matter, mind, device
        self.interactions = InteractionBuffer()
        self.tech_events: "collections.deque" = collections.deque(maxlen=256)
        self.tech_counts = torch.zeros(N_TECH, device=device)
        self._bits = torch.arange(N_TECH, device=device, dtype=torch.int32)

    def active_range(self, pool: AgentPool) -> int:
        idx = torch.nonzero(pool.alive).flatten()
        return int(idx.max().item()) + 1 if idx.numel() else 0

    def sense(self, ws: WorldState, pool: AgentPool, hi: int, nn: torch.Tensor, nd: torch.Tensor, cnt: torch.Tensor,
              ctrl: Optional[ArenaControl]) -> torch.Tensor:
        cfg, we = self.cfg, self.we
        pos, hd = pool.pos[:hi], pool.heading[:hi]
        iy, ix = we.tile_index(ws, pos)
        res = we.sample(ws.resource, iy, ix)
        gf, gl = we.gradient_local(ws.resource, iy, ix, hd)
        if ctrl is not None and ctrl.resource_visible is not None:
            vis = ctrl.resource_visible.float()
            res, gf, gl = res * vis, gf * vis, gl * vis
        hf, hl = we.gradient_local(ws.height, iy, ix, hd)
        ty = (pos[:, 1] + torch.sin(hd)).floor().long().clamp(0, ws.H - 1)
        tx = (pos[:, 0] + torch.cos(hd)).floor().long().clamp(0, ws.W - 1)
        water_ahead = (ws.is_water[ty, tx] & ~ws.filled[ty, tx]).float()
        snd = we.sample(ws.sound, iy, ix).clamp(0, 2)
        if ctrl is not None and ctrl.sound_override is not None:
            snd = ctrl.sound_override(snd)
        has = nn >= 0
        nns = nn.clamp(min=0)
        rel = pool.pos[nns] - pos
        dist = rel.norm(dim=-1).clamp(min=1e-6)
        c, s_ = torch.cos(hd), torch.sin(hd)
        prox = (1 - dist / cfg.sense_radius).clamp(0, 1) * has
        nfwd = (rel[:, 0] * c + rel[:, 1] * s_) / dist * prox
        nleft = (-rel[:, 0] * s_ + rel[:, 1] * c) / dist * prox
        cold = (0.35 - we.sample(ws.temperature, iy, ix)).clamp(min=0) * 2
        exposure = (we.sample(ws.storm, iy, ix) + cold).clamp(0, 2)
        S = torch.stack([
            torch.ones(hi, device=self.device), res, gf * 5, gl * 5, we.sample(ws.light, iy, ix),
            we.sample(ws.temperature, iy, ix), hf / 3, hl / 3, water_ahead,
            we.sample(ws.struct_h, iy, ix).float() / cfg.max_stack, we.sample(ws.enclosure, iy, ix),
            we.sample(ws.marks, iy, ix).clamp(0, 2),
            snd[:, 0], snd[:, 1], snd[:, 2], snd[:, 3],
            (cnt / 8).clamp(0, 1), nfwd, nleft, pool.color[nns, 0] * has, pool.color[nns, 1] * has,
            pool.speed[nns] * has, (pool.carried[nns] > 0).float() * has,
            pool.energy[:hi], pool.integrity[:hi], pool.fatigue[:hi], (pool.last_damage[:hi] * 20).clamp(0, 1),
            pool.carried[:hi] / cfg.carry_capacity, pool.contact[:hi] + pool.blocked[:hi], exposure], 1)
        pool.sense_cache[:hi] = S
        return S

    @staticmethod
    def _slice_ctrl(ctrl: ArenaControl, hi: int) -> ArenaControl:
        """Controles de laboratório têm uma linha por slot; o passo usa só [:hi]."""
        cut = lambda t: t[:hi] if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] > hi else t
        c = dc_replace(ctrl, **{f.name: cut(getattr(ctrl, f.name)) for f in dc_fields(ctrl)
                                if isinstance(getattr(ctrl, f.name), torch.Tensor)})
        if ctrl.iv is not None:
            c.iv = dc_replace(ctrl.iv, **{f.name: cut(getattr(ctrl.iv, f.name)) for f in dc_fields(ctrl.iv)
                                          if isinstance(getattr(ctrl.iv, f.name), torch.Tensor)})
        return c

    def step(self, ws: WorldState, pool: AgentPool, tick: int, gen: torch.Generator, lineage: Optional[LineageStore],
             ids: IdCounter, ledger: StructureLedger, bodies: Optional[Dict[int, BodyRegistry]] = None,
             ctrl: Optional[ArenaControl] = None, record_interactions: bool = True,
             hi_override: Optional[int] = None) -> Dict[str, float]:
        cfg, d, L = self.cfg, self.device, self.L
        hi = hi_override if hi_override is not None else self.active_range(pool)
        if ctrl is not None:
            ctrl = self._slice_ctrl(ctrl, hi)
        stats = {"alive": 0, "births": 0, "deaths": 0, "eaten": 0.0, "placed": 0, "picked": 0, "given": 0, "taken": 0,
                 "deaths_starvation": 0, "deaths_exposure": 0, "deaths_age": 0, "deaths_conflict": 0}
        if hi == 0:
            self.we.sound_step(ws)
            return stats
        alive = pool.alive[:hi]
        pos, hd = pool.pos[:hi], pool.heading[:hi]
        nn, nd, cnt = spatial_neighbors(pos, alive, cfg.sense_radius)
        S = self.sense(ws, pool, hi, nn, nd, cnt, ctrl)
        out = self.mind.step(pool, hi, S, gen, ctrl.iv if ctrl is not None else None)
        if ctrl is not None and ctrl.action_block is not None:
            # canais contínuos (forward/turn) ablados → 0; portas ablados → fechadas
            fill = torch.full_like(out, -8.0); fill[:, :2] = 0.0
            out = torch.where(ctrl.action_block, fill, out)
        if ctrl is not None and ctrl.action_force is not None:
            out = torch.where(ctrl.action_force, torch.full_like(out, 8.0), out)
        act = torch.sigmoid(out)
        fwd = torch.tanh(out[:, 0]); turn = torch.tanh(out[:, 1])
        rest = act[:, 15] > 0.5
        G = pool.genome[:hi]
        # ---------------- movimento: rastejo basal comum a todos os níveis + termo do corpo
        # F1: termo do corpo = resumo físico medido da marcha do genoma (velocidade e giro sob comando padrão);
        # F2+: termo do corpo = deslocamento e guinada calculados pela física do próprio corpo neste tick.
        # O mesmo ganho corpo→mundo (cfg.body_world_scale, coarse-graining espaço-temporal declarado) vale nos dois.
        cap = pool.body_cap[:hi]
        fwd_eff = torch.where(fwd >= 0, fwd, 0.3 * fwd) * (~rest) * (1 - 0.5 * pool.fatigue[:hi])
        vbody = (cap[:, 0] * cfg.body_world_scale).clamp(max=cfg.max_speed_cap - cfg.min_speed) * fwd_eff
        crawl = cfg.min_speed * fwd_eff
        new_h = hd + turn * (self.BASE_TURN + cap[:, 3].clamp(max=self.MAX_BODY_TURN))
        gait = new_h + cap[:, 4]                         # direção líquida da marcha no referencial do corpo (medida)
        dx = torch.cos(new_h) * crawl + torch.cos(gait) * vbody
        dy = torch.sin(new_h) * crawl + torch.sin(gait) * vbody
        # coarse-graining deriva-difusão: a oscilação da marcha medida na física vira termo difusivo (gerador semeado)
        wob = (cap[:, 5] * cfg.body_world_scale).clamp(max=self.MAX_WOBBLE) * alive
        noise = torch.randn(hi, 2, generator=gen, device=d) * wob.unsqueeze(1)
        dx = dx + noise[:, 0]; dy = dy + noise[:, 1]
        contact = torch.zeros(hi, device=d)
        if bodies:
            for D_, reg in bodies.items():
                act_b = reg.agent_slot >= 0
                if not bool(act_b.any()):
                    continue
                aslot = reg.agent_slot.clamp(min=0, max=hi - 1)
                ok_b = act_b & (reg.agent_slot < hi)
                motor = torch.stack([fwd_eff[aslot], turn[aslot]], 1) * ok_b.unsqueeze(1)
                r = reg.step(motor)
                bi = torch.nonzero(ok_b & (reg.agent_slot >= 0)).flatten()
                if bi.numel():
                    a_i = reg.agent_slot[bi]
                    db = r["disp"][bi, :2] * cfg.body_world_scale
                    mag = db.norm(dim=-1, keepdim=True)
                    db = db * (mag.clamp(max=cfg.max_speed_cap - cfg.min_speed + self.MAX_WOBBLE) / mag.clamp(min=1e-9))
                    h0 = hd[a_i] + turn[a_i] * self.BASE_TURN + r["dyaw"][bi].clamp(-self.MAX_BODY_TURN, self.MAX_BODY_TURN)
                    # eixo x do referencial da física do corpo, no mundo: rumo − guinada acumulada do corpo
                    ang = h0 - reg.bb.yaw[bi]
                    ca, sa = torch.cos(ang), torch.sin(ang)
                    c0, s0 = torch.cos(h0), torch.sin(h0)
                    cr = cfg.min_speed * fwd_eff[a_i]
                    dx[a_i] = db[:, 0] * ca - db[:, 1] * sa + cr * c0
                    dy[a_i] = db[:, 0] * sa + db[:, 1] * ca + cr * s0
                    new_h[a_i] = h0
                    contact[a_i] = r["contact"][bi]
        pool.contact[:hi] = contact
        if ctrl is not None and ctrl.freeze_move is not None:
            dx = dx * (~ctrl.freeze_move); dy = dy * (~ctrl.freeze_move)
        nx = (pos[:, 0] + dx).clamp(0, ws.W - 1e-3); ny = (pos[:, 1] + dy).clamp(0, ws.H - 1e-3)
        tiy, tix = ny.floor().long(), nx.floor().long()
        iy, ix = self.we.tile_index(ws, pos)
        walk = ws.walkable()[tiy, tix]
        climb = (ws.height[tiy, tix] - ws.height[iy, ix]).abs() <= 2.5
        moved = walk & climb & alive
        pool.blocked[:hi] = (~moved & alive & ((dx.abs() + dy.abs()) > 1e-4)).float()
        newpos = torch.where(moved.unsqueeze(1), torch.stack([nx, ny], 1), pos)
        dist_moved = (newpos - pos).norm(dim=-1)
        pool.pos[:hi] = newpos
        pool.heading[:hi] = torch.remainder(new_h, 2 * math.pi)
        pool.speed[:hi] = dist_moved
        iy, ix = self.we.tile_index(ws, newpos)
        tile = iy * ws.W + ix
        tbits = torch.zeros(hi, dtype=torch.int32, device=d)
        tbits |= (moved & ws.filled[iy, ix] & ws.is_water[iy, ix]).int() << 7
        # ---------------- consumo (manchas ricas: extração plena só com ≥2 consumidores simultâneos)
        eat = (act[:, 6] > 0.5) & alive
        gained = torch.zeros(hi, device=d)
        if bool(eat.any()):
            et = tile[eat]
            ccount = torch.zeros(ws.H * ws.W, device=d).index_add_(0, et, torch.ones(et.numel(), device=d))
            resf = ws.resource.flatten()
            demand = ccount[et] * cfg.eat_rate
            frac = torch.clamp(resf[et] / demand.clamp(min=1e-9), max=1.0)
            amount = cfg.eat_rate * frac
            richm = ws.rich.flatten()[et]
            eff = torch.where(richm, torch.where(ccount[et] >= 2, torch.full_like(amount, 1.5), torch.full_like(amount, 0.3)),
                              torch.ones_like(amount))
            used = torch.zeros(ws.H * ws.W, device=d).index_add_(0, et, amount)
            ws.resource = (resf - used).clamp(min=0).view(ws.H, ws.W)
            gained[eat] = amount * eff * cfg.eat_efficiency
            stats["eaten"] = float(amount.sum())
            tbits[eat] |= ((richm & (ccount[et] >= 2)).int() << 2)
        energy = pool.energy[:hi] + gained
        cost = cfg.basal_cost * (1 + cap[:, 2] / 64) + cfg.move_cost * dist_moved * (1 + cap[:, 1].clamp(max=50) / 50)
        # ---------------- matéria
        pick = (act[:, 7] > 0.5) & alive & (pool.carried[:hi] < cfg.carry_capacity)
        if bool(pick.any()):
            pi = torch.nonzero(pick).flatten()
            ok = self.matter.pick(ws, ledger, pi, newpos[pi], pool.heading[pi], pool.carried[pi])
            got = pi[ok]
            pool.carried[got] += 1
            cost[pi] += cfg.action_cost * 2
            stats["picked"] = int(ok.sum())
            pool.stats[got, 3] += 1
        place = (act[:, 8] > 0.5) & alive & (pool.carried[:hi] >= 1)
        if bool(place.any()):
            pl = torch.nonzero(place).flatten()
            aty, atx = self.matter.ahead_tile(ws, newpos[pl], pool.heading[pl])
            wat = ws.is_water[aty, atx] & ~ws.filled[aty, atx]
            ok = self.matter.place(ws, ledger, newpos[pl], pool.heading[pl], pool.aid[pl], tick)
            done = pl[ok]
            tbits[pl] |= (ok & wat).int() << 0
            tbits[pl] |= (ok & ~wat & (ws.struct_h[aty, atx] >= 2)).int() << 1
            pool.carried[done] -= 1
            cost[done] += cfg.build_cost
            stats["placed"] = int(ok.sum())
            pool.stats[done, 2] += 1
        mark = (act[:, 10] > 0.5) & alive
        if bool(mark.any()):
            mk = ws.marks.flatten().index_add_(0, tile[mark], torch.full((int(mark.sum()),), 0.5, device=d))
            ws.marks = mk.clamp(max=4).view(ws.H, ws.W)
            cost += mark.float() * cfg.action_cost
            tbits |= mark.int() << 4
        # ---------------- sinais emitidos (canal físico de som) e exibição de cor
        emit = act[:, 2:6] * alive.unsqueeze(1)
        if ctrl is not None and ctrl.mute is not None:
            emit = emit * (~ctrl.mute).unsqueeze(1)
        pool.emit[:hi] = emit
        loud = emit.sum(1) > 0.05
        if bool(loud.any()):
            snd = ws.sound.view(cfg.n_sound, -1)
            snd.index_add_(1, tile[loud], emit[loud].T.contiguous())
        cost += cfg.signal_cost * (emit ** 2).sum(1)
        pool.color[:hi] = 0.7 * pool.color[:hi] + 0.3 * act[:, 13:15]
        # ---------------- interação social (vizinho mais próximo)
        has = (nn >= 0) & alive
        nns = nn.clamp(min=0)
        conflict_hit = torch.zeros(hi, device=d)
        give = (act[:, 11] > 0.5) & has
        take = (act[:, 12] > 0.5) & has & ~give
        if bool(give.any()):
            gi = torch.nonzero(give).flatten()
            amt = torch.minimum(torch.full((gi.numel(),), cfg.transfer_amount, device=d), energy[gi] * 0.2)
            energy[gi] -= amt
            energy.index_add_(0, nns[gi], amt * 0.95)
            cost[gi] += cfg.action_cost
            pool.stats[gi, 4] += amt
            stats["given"] = int(gi.numel())
            tbits[gi] |= 1 << 3
            if record_interactions:
                self.interactions.push(pool.aid[gi], pool.aid[nns[gi]], 1, amt, tick)
        if bool(take.any()):
            ti = torch.nonzero(take).flatten()
            tgt = nns[ti]
            win = pool.integrity[ti] >= 0.8 * pool.integrity[tgt]
            amt = torch.minimum(torch.full((ti.numel(),), cfg.transfer_amount, device=d), energy[tgt].clamp(min=0) * 0.2) * win
            energy.index_add_(0, tgt, -amt)
            energy[ti] += amt * 0.9
            hit = torch.zeros(hi, device=d).index_add_(0, tgt, 0.01 * win.float())
            pool.integrity[:hi] -= hit
            pool.last_damage[:hi] += hit
            conflict_hit += hit
            cost[ti] += cfg.action_cost * 2
            pool.stats[ti, 5] += amt
            stats["taken"] = int(ti.numel())
            tbits[ti] |= ((amt > 0).int() << 5)
            if record_interactions:
                self.interactions.push(pool.aid[ti], pool.aid[tgt], 2, amt, tick)
        if record_interactions:
            close = has & (nd < cfg.sense_radius * 0.5)
            if bool(close.any()):
                ci = torch.nonzero(close).flatten()
                self.interactions.push(pool.aid[ci], pool.aid[nns[ci]], 0, torch.full((ci.numel(),), 0.05, device=d), tick)
                pool.last_partner[ci] = pool.aid[nns[ci]]
        tbits |= (rest & (ws.enclosure[iy, ix] >= 0.5) & alive).int() << 6
        tbits = tbits * alive.int()
        newb = tbits & ~pool.tech[:hi]
        if bool((newb != 0).any()):
            ni = torch.nonzero(newb).flatten()
            self.tech_events.append((pool.aid[ni].clone(), newb[ni].clone(), pool.last_partner[ni].clone(), tick,
                                     newpos[ni].clone()))
        pool.tech[:hi] |= tbits
        self.tech_counts += ((tbits.unsqueeze(1) >> self._bits) & 1).sum(0).float()
        # ---------------- metabolismo, exposição, abrigo, reparo
        cost += cfg.action_cost * 0.5 * (act[:, 6] > 0.5).float()
        energy = energy - cost * alive
        temp = ws.temperature[iy, ix]
        exposure = ws.storm[iy, ix] + (0.35 - temp).clamp(min=0) * 2
        shelter = ws.enclosure[iy, ix]
        dmg = 0.006 * exposure * (1 - shelter)
        if ctrl is not None and ctrl.damage is not None:
            dmg = dmg + ctrl.damage
        starving = energy < 0
        dmg = dmg + starving.float() * 0.02
        energy = energy.clamp(min=0)
        heal = (energy > 0.5).float() * 0.002
        energy = energy - heal * 0.5
        if ctrl is not None and ctrl.energy_floor is not None:
            energy = energy.clamp(min=ctrl.energy_floor)
        integ = (pool.integrity[:hi] - dmg + heal).clamp(max=1.0)
        pool.last_damage[:hi] = 0.8 * pool.last_damage[:hi] + dmg
        pool.energy[:hi] = torch.where(alive, energy.clamp(max=1.0), pool.energy[:hi])
        pool.integrity[:hi] = torch.where(alive, integ, pool.integrity[:hi])
        fat = pool.fatigue[:hi] + 0.02 * dist_moved + 0.005 * (act[:, 6:13] > 0.5).float().sum(1) - 0.04 * rest.float() - 0.002
        pool.fatigue[:hi] = torch.where(alive, fat.clamp(0, 1), pool.fatigue[:hi])
        pool.age[:hi] += alive.float()
        pool.repro_cool[:hi] = (pool.repro_cool[:hi] - 1).clamp(min=0)
        pool.stats[:hi, 0] += gained
        pool.stats[:hi, 1] += dist_moved
        pool.stats[:hi, 7] += dmg
        # assinatura comportamental (para cultura/especialização): média móvel das ações
        avec = torch.cat([torch.stack([fwd, turn], 1), (act[:, 2:] > 0.5).float()], 1)
        pool.signature[:hi] = torch.where(alive.unsqueeze(1), 0.98 * pool.signature[:hi] + 0.02 * avec, pool.signature[:hi])
        # tráfego (infraestrutura emergente: trilhas usadas repetidamente)
        tr = ws.traffic.flatten().index_add_(0, tile[alive & (dist_moved > 0.01)],
                                              torch.full((int((alive & (dist_moved > 0.01)).sum()),), 0.01, device=d))
        ws.traffic = tr.clamp(max=1).view(ws.H, ws.W)
        # ---------------- reprodução
        if ctrl is None or not ctrl.no_repro:
            stats["births"] = self._reproduce(ws, pool, hi, act, nn, tick, gen, lineage, ids)
        # ---------------- morte funcional
        if ctrl is None or not ctrl.no_death:
            lifespan = L.view(G, "life_span")
            dead_int = alive & (pool.integrity[:hi] <= 0)
            dead_age = alive & (pool.age[:hi] > lifespan)
            dead = dead_int | dead_age
            if bool(dead.any()):
                starv = dead_int & (pool.energy[:hi] <= 0.001)
                confl = dead_int & ~starv & (conflict_hit > 0)
                stats["deaths"] = int(dead.sum())
                stats["deaths_age"] = int((dead_age & ~dead_int).sum())
                stats["deaths_starvation"] = int(starv.sum())
                stats["deaths_conflict"] = int(confl.sum())
                stats["deaths_exposure"] = int((dead_int & ~starv & ~confl).sum())
                self.kill(pool, torch.nonzero(dead).flatten(), bodies, ws)
        self.we.sound_step(ws)
        stats["alive"] = int(pool.alive.sum())
        return stats

    def _reproduce(self, ws, pool, hi, act, nn, tick, gen, lineage, ids) -> int:
        cfg, L, d = self.cfg, self.L, self.device
        G = pool.genome[:hi]
        want = (act[:, 9] > 0.5) & pool.alive[:hi] & (pool.energy[:hi] > L.view(G, "life_repro")) \
            & (pool.age[:hi] > L.view(G, "life_maturity")) & (pool.repro_cool[:hi] <= 0)
        par = torch.nonzero(want).flatten()
        if par.numel() == 0:
            return 0
        free = pool.free_slots(par.numel())
        par = par[: free.numel()]
        n = par.numel()
        if n == 0:
            return 0
        child_g = L.mutate(pool.genome[par], gen)
        mate = nn[par]
        sexual = (mate >= 0)
        if bool(sexual.any()):
            mi = mate[sexual]
            mixed = L.crossover(child_g[sexual], L.mutate(pool.genome[mi], gen), gen)
            child_g[sexual] = mixed
        off = (torch.rand(n, 2, generator=gen, device=d) - 0.5) * 2.0
        cpos = (pool.pos[par] + off)
        cpos[:, 0] = cpos[:, 0].clamp(0, ws.W - 1e-3); cpos[:, 1] = cpos[:, 1].clamp(0, ws.H - 1e-3)
        ciy, cix = cpos[:, 1].floor().long(), cpos[:, 0].floor().long()
        cpos = torch.where(ws.walkable()[ciy, cix].unsqueeze(1), cpos, pool.pos[par])
        e_child = pool.energy[par] * 0.45
        pool.energy[par] = pool.energy[par] * (1 - cfg.repro_cost_fraction)
        pool.repro_cool[par] = 50
        pool.stats[par, 6] += 1
        aids = ids.take(n)
        gens = (pool.generation[par] + 1).cpu().numpy()
        parents = pool.aid[par].cpu().numpy()
        roots = pool.root[par].cpu().numpy()
        pool.spawn(free, child_g, cpos, aids, aids, parents, roots, gens, tick, e_child, body_cap=pool.body_cap[par],
                   species=pool.species[par], gen=gen)
        if lineage is not None:
            lineage.add(aids, parents, tick, "birth", gens)
        return n

    def kill(self, pool: AgentPool, slots: torch.Tensor, bodies: Optional[Dict[int, BodyRegistry]] = None,
             ws: Optional[WorldState] = None):
        """Morte funcional (ou absorção em F0): a matéria carregada cai no tile (conservação de matéria)."""
        if slots.numel() == 0:
            return
        if ws is not None:
            carried = pool.carried[slots]
            if bool((carried > 0).any()):
                iy, ix = WorldEngine.tile_index(ws, pool.pos[slots])
                dep = ws.deposit.flatten().index_add_(0, iy * ws.W + ix, carried)
                ws.deposit = dep.view(ws.H, ws.W)
            pool.carried[slots] = 0
        bs = pool.body_slot[slots]
        if bodies and bool((bs >= 0).any()):
            for D_, reg in bodies.items():
                hit = torch.nonzero(torch.isin(reg.agent_slot, slots)).flatten()
                reg.detach(hit)
        pool.alive[slots] = False
        pool.body_slot[slots] = -1
        pool.fidelity[slots] = 0

    def spawn_founders(self, ws: WorldState, pool: AgentPool, n: int, gen: torch.Generator, ids: IdCounter, tick: int,
                       lineage: Optional[LineageStore], genomes: Optional[torch.Tensor] = None, origin: str = "founder",
                       region: Optional[Tuple[int, int, int, int]] = None, energy: float = 0.8) -> torch.Tensor:
        d = self.device
        free = pool.free_slots(n)
        n = free.numel()
        if n == 0:
            return free
        walk = ws.walkable() & (ws.fertility > 0.05) if ws.kind == "planet" else ws.walkable()
        if region is not None:
            y0, x0, y1, x1 = region
            m = torch.zeros_like(walk); m[y0:y1, x0:x1] = True
            walk = walk & m
        cand = torch.nonzero(walk)
        if cand.shape[0] == 0:
            cand = torch.nonzero(ws.walkable())
        pick = cand[torch.randint(0, cand.shape[0], (n,), generator=gen, device=d)]
        pos = torch.stack([pick[:, 1].float() + torch.rand(n, generator=gen, device=d),
                           pick[:, 0].float() + torch.rand(n, generator=gen, device=d)], 1)
        G = genomes[:n] if genomes is not None else self.L.random(n, gen)
        aids = ids.take(n)
        zeros = np.zeros(n, np.int64)
        pool.spawn(free, G, pos, aids, aids, zeros, aids, zeros, tick, torch.full((n,), energy, device=d), gen=gen)
        if lineage is not None:
            lineage.add(aids, zeros, tick, origin, zeros)
        return free
