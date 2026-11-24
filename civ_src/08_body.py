# %% [markdown]
# ## 08 — BODY ENGINE: Matter Nodes em sistema massa-mola 3D (lote na GPU)
# Cada corpo F2+ é um grafo de **Matter Nodes** (posição, velocidade, massa, raio, propriedades contínuas, estado de
# sinal, estado de desenvolvimento, linhagem da célula) ligados por molas (vizinhança 6 + diagonais de face).
# Física artificial simplificada: gravidade, contato com o solo por penalidade, atrito de Coulomb suavizado
# (coeficiente da adesão do nó), molas com amortecimento, arrasto, atuação por encurtamento do comprimento de repouso
# (contração × inervação × comandos motores + oscilador interno com fase do morfógeno). Locomoção não é programada:
# emerge (ou não) do padrão de atuação e do atrito.
# Sinais internos: estado de sinal por nó propagado pela condutividade (difusão no grafo) e excitado por contato e
# atuação — base da câmera de sinais e dos sensores proprioceptivos.

# %%
class BodyBatch:
    FIELDS = ["pos", "vel", "active", "mass", "k_e", "c_e", "rest", "e_on", "gain", "innerv", "phase", "mu", "cond",
              "sens", "signal", "osc", "t", "yaw", "props", "parent_cell", "birth_step"]

    def __init__(self, eng: "BodyEngine", bp: Blueprint):
        d, dev = eng, eng.device
        occ = bp.occ
        B, N = occ.shape
        P = bp.props
        self.B, self.N = B, N
        grid = d.dev.grid.float()
        pos = grid.unsqueeze(0).expand(B, N, 3).clone()
        w = occ.float().unsqueeze(-1)
        com = (pos * w).sum(1) / w.sum(1).clamp(min=1)
        pos[..., 0] -= com[:, 0:1]; pos[..., 1] -= com[:, 1:2]
        zmin = torch.where(occ, pos[..., 2], torch.full_like(pos[..., 2], 1e9)).min(1).values
        pos[..., 2] -= zmin.unsqueeze(1) - 0.05
        self.pos = pos * occ.unsqueeze(-1)
        self.vel = torch.zeros_like(self.pos)
        self.active = occ
        self.mass = (0.4 + 1.2 * P[..., 9]) * occ + (~occ) * 1.0
        e0, e1 = d.dev.edges[:, 0], d.dev.edges[:, 1]
        self.e_on = occ[:, e0] & occ[:, e1]
        stiff = 0.5 * (P[:, e0, 0] + P[:, e1, 0])
        damp = 0.5 * (P[:, e0, 1] + P[:, e1, 1])
        self.k_e = (d.k_base * (0.25 + stiff)) * self.e_on
        self.c_e = (d.c_base * (0.2 + damp)) * self.e_on
        self.rest = d.dev.edge_rest.unsqueeze(0).expand(B, -1).clone()
        self.gain = 0.35 * P[..., 3] * occ
        self.innerv = bp.innerv
        self.phase = bp.phase
        self.mu = (0.15 + 1.1 * P[..., 2]) * occ
        self.cond = P[..., 4] * occ
        self.sens = P[..., 5] * occ
        self.signal = torch.zeros(B, N, device=dev)
        self.osc = bp.osc.clone()
        self.t = torch.zeros(B, device=dev)
        self.yaw = torch.zeros(B, device=dev)
        self.props = P
        self.parent_cell = bp.parent_cell
        self.birth_step = bp.birth_step

    def com(self) -> torch.Tensor:
        w = self.active.float().unsqueeze(-1)
        return (self.pos * w).sum(1) / w.sum(1).clamp(min=1)

    def select(self, idx: torch.Tensor) -> "BodyBatch":
        out = BodyBatch.__new__(BodyBatch)
        for f in self.FIELDS:
            setattr(out, f, getattr(self, f)[idx].clone())
        out.B, out.N = int(idx.numel()), self.N
        return out


class BodyEngine:
    def __init__(self, cfg: CivConfig, dev_engine: DevelopmentEngine, device: torch.device):
        self.cfg, self.dev, self.device = cfg, dev_engine, device
        self.k_base, self.c_base = 40.0, 1.2
        self.g, self.k_ground, self.c_ground, self.drag = 9.8, 400.0, 8.0, 0.05

    def make(self, bp: Blueprint) -> BodyBatch:
        return BodyBatch(self, bp)

    def step(self, bb: BodyBatch, motor: torch.Tensor, dt: Optional[float] = None, substeps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """motor [B,2] = (forward, turn) do controlador; o oscilador interno é o terceiro canal motor."""
        dt = dt or self.cfg.body_dt
        substeps = substeps or self.cfg.body_substeps
        e0, e1 = self.dev.edges[:, 0], self.dev.edges[:, 1]
        act_n = bb.active.float()
        com0 = bb.com()
        work = torch.zeros(bb.B, device=self.device)
        contact_frac = torch.zeros(bb.B, device=self.device)
        strain_mean = torch.zeros(bb.B, device=self.device)
        for _ in range(substeps):
            osc = torch.sin(2 * math.pi * bb.osc.unsqueeze(1) * bb.t.unsqueeze(1) + bb.phase)
            drive = (bb.innerv[..., 0] * motor[:, 0:1] + bb.innerv[..., 1] * motor[:, 1:2] + bb.innerv[..., 2] * osc)
            a = bb.gain * torch.tanh(drive) * act_n
            d = bb.pos[:, e1] - bb.pos[:, e0]
            Ln = d.norm(dim=-1) + 1e-6
            u = d / Ln.unsqueeze(-1)
            act_e = 0.5 * (a[:, e0] + a[:, e1])
            rest_t = bb.rest * (1 - act_e)
            vrel = ((bb.vel[:, e1] - bb.vel[:, e0]) * u).sum(-1)
            f = (bb.k_e * (Ln - rest_t) + bb.c_e * vrel)
            Fv = torch.zeros_like(bb.pos)
            Fv.index_add_(1, e0, f.unsqueeze(-1) * u)
            Fv.index_add_(1, e1, -f.unsqueeze(-1) * u)
            Fv[..., 2] -= bb.mass * self.g
            pen = (-bb.pos[..., 2]).clamp(min=0)
            incontact = (pen > 0).float()
            Fn = (self.k_ground * pen - self.c_ground * bb.vel[..., 2] * incontact).clamp(min=0)
            Fv[..., 2] += Fn
            vt = bb.vel[..., :2]
            vtn = vt.norm(dim=-1, keepdim=True)
            Fv[..., :2] -= (bb.mu * Fn).unsqueeze(-1) * vt / (vtn + 0.05)
            Fv -= self.drag * bb.vel
            bb.vel = (bb.vel + dt * Fv / bb.mass.unsqueeze(-1)) * act_n.unsqueeze(-1)
            bb.pos = bb.pos + dt * bb.vel
            bb.t = bb.t + dt
            work += ((bb.k_e * (Ln - rest_t)).abs() * act_e.abs()).sum(1) * dt
            contact_frac += (incontact * act_n).sum(1) / act_n.sum(1).clamp(min=1)
            strain_mean += (((Ln - bb.rest) / bb.rest).abs() * bb.e_on).sum(1) / bb.e_on.sum(1).clamp(min=1)
        # sinais internos: excitação por contato e atuação, propagação pela condutividade (difusão no grafo)
        nbsig = bb.signal[:, self.dev.nb_safe] * (self.dev.nb_valid.unsqueeze(0) & bb.active[:, self.dev.nb_safe])
        mean_nb = nbsig.sum(-1) / (self.dev.nb_valid.unsqueeze(0) & bb.active[:, self.dev.nb_safe]).sum(-1).clamp(min=1)
        excite = bb.sens * ((-bb.pos[..., 2]).clamp(min=0) * 10 + a.abs())
        bb.signal = ((bb.signal + bb.cond * (mean_nb - bb.signal) + excite) * 0.85 * act_n).clamp(0, 4)
        com1 = bb.com()
        disp = com1 - com0
        # guinada (yaw) do corpo: rotação média dos nós em torno do COM no plano xy
        rel = (bb.pos[..., :2] - com1[:, None, :2])
        ang = torch.atan2(rel[..., 1], rel[..., 0])
        if not hasattr(bb, "_ang_prev"):
            bb._ang_prev = ang
        dang = torch.atan2(torch.sin(ang - bb._ang_prev), torch.cos(ang - bb._ang_prev))
        r2 = (rel ** 2).sum(-1) * act_n                       # nós perto do COM têm ângulo ruidoso: peso ∝ r²
        dyaw = (dang * r2).sum(1) / r2.sum(1).clamp(min=1e-6)
        bb._ang_prev = ang
        bb.yaw = bb.yaw + dyaw
        # recentra horizontalmente (invariância de translação; o mundo guarda a posição)
        bb.pos[..., 0] -= com1[:, 0:1] * act_n; bb.pos[..., 1] -= com1[:, 1:2] * act_n
        bb._ang_prev = torch.atan2(bb.pos[..., 1] - 0, bb.pos[..., 0] - 0)
        return {"disp": disp, "dyaw": dyaw, "work": work, "contact": contact_frac / substeps,
                "strain": strain_mean / substeps, "com_z": com1[:, 2], "finite": torch.isfinite(bb.pos).all(-1).all(-1)}

    def measure_capability(self, G: torch.Tensor, seeds: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Teste padronizado de marcha (comando 'para frente' constante + oscilador) e de giro → resumo usado pelo
        agente leve F1: [velocidade, custo por tick, nº de nós, taxa de giro]. Física real, frequência baixa."""
        return self.measure_blueprint(self.dev.develop(G, seeds), steps)

    def measure_blueprint(self, bp: Blueprint, steps: Optional[int] = None) -> torch.Tensor:
        steps = steps or max(8, self.cfg.probe.body_test_steps // self.cfg.body_substeps)
        bb = self.make(bp)
        fwd = torch.tensor([1.0, 0.0], device=self.device).expand(bb.B, 2)
        net = torch.zeros(bb.B, 2, device=self.device); work = torch.zeros(bb.B, device=self.device)
        sq = torch.zeros(bb.B, device=self.device)
        for i in range(steps):
            r = self.step(bb, fwd)
            if i >= steps // 4:
                # deslocamento no referencial do próprio corpo (desfaz a guinada acumulada)
                c, s_ = torch.cos(-bb.yaw), torch.sin(-bb.yaw)
                d2 = r["disp"][:, :2]
                db = torch.stack([d2[:, 0] * c - d2[:, 1] * s_, d2[:, 0] * s_ + d2[:, 1] * c], 1)
                net += db
                sq += (db ** 2).sum(-1)
                work += r["work"]
        bb2 = self.make(bp)
        trn = torch.tensor([0.0, 1.0], device=self.device).expand(bb2.B, 2)
        for _ in range(steps // 2):
            r2 = self.step(bb2, trn)
        meas = max(1, steps - steps // 4)
        net = torch.nan_to_num(net / meas)
        speed = net.norm(dim=-1)                              # progresso LÍQUIDO por tick (desconta oscilação)
        wobble = torch.nan_to_num((sq / meas - speed ** 2).clamp(min=0).sqrt())   # desvio-padrão por tick (difusão)
        gait = torch.atan2(net[:, 1], net[:, 0])               # direção da marcha relativa ao eixo do corpo
        cost = torch.nan_to_num(work / meas)
        turn = torch.nan_to_num(bb2.yaw.abs() / max(1, steps // 2))
        n = bp.occ.sum(1).float()
        return torch.stack([speed, cost, n, turn, torch.nan_to_num(gait), wobble], 1)
