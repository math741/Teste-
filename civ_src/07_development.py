# %% [markdown]
# ## 07 — DEVELOPMENT ENGINE: semente → corpo
# O `DevelopmentGenome` guarda **regras locais**, não o corpo final. Em lote (B genomas) numa grade 3D D³:
# 1. célula-semente com morfógenos iniciais; campo materno de polaridade (eixo do genoma);
# 2. por passo: difusão entre células ocupadas + reação regulatória (matriz do genoma, tanh) + decaimento;
# 3. crescimento: sítios vizinhos vazios com sinal de crescimento > limiar (ruído de desenvolvimento determinístico por
#    semente) e orçamento de células; a célula nova herda morfógenos dos vizinhos e registra a célula-mãe
#    (mapa de destino / linhagem das unidades);
# 4. leitura: propriedades contínuas da matéria (sigmoide), inervação motora e fase de oscilação.
# Nada nasce "osso", "músculo" ou "nervo": regiões podem adquirir propriedades funcionais.

# %%
@dataclass
class Blueprint:
    occ: torch.Tensor          # [B,N] bool
    props: torch.Tensor        # [B,N,P]
    morph: torch.Tensor        # [B,k,N]
    innerv: torch.Tensor       # [B,N,N_MOTOR]
    phase: torch.Tensor        # [B,N]
    osc: torch.Tensor          # [B]
    parent_cell: torch.Tensor  # [B,N] int (-1 = semente/vazio)
    birth_step: torch.Tensor   # [B,N] int
    frames: Optional[List[torch.Tensor]] = None

    def n_nodes(self) -> torch.Tensor:
        return self.occ.sum(1)


class DevelopmentEngine:
    def __init__(self, cfg: CivConfig, layout: GenomeLayout, device: torch.device):
        self.cfg, self.L, self.device = cfg, layout, device
        D = cfg.dev_lattice
        self.D, self.N = D, D ** 3
        g = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(D), torch.arange(D), indexing="ij"), -1).reshape(-1, 3)
        self.grid = g.to(device)                                       # (x,y,z) inteiros
        self.coords = (g.float() / max(1, D - 1) * 2 - 1).to(device)  # [-1,1]^3
        idx = lambda x, y, z: (x * D + y) * D + z
        nb = torch.full((self.N, 6), -1, dtype=torch.long)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i, (x, y, z) in enumerate(g.tolist()):
            for j, (dx, dy, dz) in enumerate(dirs):
                a, b, c = x + dx, y + dy, z + dz
                if 0 <= a < D and 0 <= b < D and 0 <= c < D:
                    nb[i, j] = idx(a, b, c)
        self.nb = nb.to(device)
        self.nb_valid = self.nb >= 0
        self.nb_safe = self.nb.clamp(min=0)
        edges, rest = [], []
        offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
        for i, (x, y, z) in enumerate(g.tolist()):
            for dx, dy, dz in offs:
                a, b, c = x + dx, y + dy, z + dz
                if 0 <= a < D and 0 <= b < D and 0 <= c < D:
                    edges.append((i, idx(a, b, c))); rest.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        self.edges = torch.tensor(edges, dtype=torch.long, device=device)
        self.edge_rest = torch.tensor(rest, dtype=torch.float32, device=device)
        c = D // 2
        self.seed_cell = idx(c, c, min(1, D - 1))
        self.cell_idx = torch.arange(self.N, device=device, dtype=torch.float32)

    @staticmethod
    def hnoise(seeds: torch.Tensor, step: int, cells: torch.Tensor) -> torch.Tensor:
        """Ruído determinístico por (semente, passo, célula), independente da composição do lote."""
        s = (seeds % 100003).float().unsqueeze(1)
        x = s * 12.9898 + cells.unsqueeze(0) * 78.233 + step * 37.719
        return torch.frac(torch.sin(x) * 43758.5453) * 2 - 1

    def develop(self, G: torch.Tensor, seeds: torch.Tensor, record: bool = False, noise_scale: float = 1.0) -> Blueprint:
        L, B, N, k = self.L, G.shape[0], self.N, self.cfg.n_morph
        d = self.device
        Wm, bm = L.view(G, "dev_Wm"), L.view(G, "dev_bm")
        decay, diff, seedm = L.view(G, "dev_decay"), L.view(G, "dev_diff"), L.view(G, "dev_seed")
        axis, gw, gthr = L.view(G, "dev_axis"), L.view(G, "dev_gw"), L.view(G, "dev_gthr")
        maxn = (L.view(G, "dev_maxfrac") * N).ceil().clamp(min=1)
        dnoise = L.view(G, "dev_noise") * noise_scale
        occ = torch.zeros(B, N, dtype=torch.bool, device=d); occ[:, self.seed_cell] = True
        m = torch.zeros(B, k, N, device=d); m[:, :, self.seed_cell] = seedm
        pol = axis @ self.coords.T                                    # [B,N] campo materno de polaridade
        parent = torch.full((B, N), -1, dtype=torch.long, device=d)
        birth = torch.full((B, N), -1, dtype=torch.long, device=d); birth[:, self.seed_cell] = 0
        frames = [occ.clone()] if record else None
        valid = self.nb_valid.unsqueeze(0)
        for g in range(self.cfg.dev_steps):
            occf = occ.float()
            nocc = occf[:, self.nb_safe] * valid                     # [B,N,6]
            nbm = m[:, :, self.nb_safe] * nocc.unsqueeze(1)          # [B,k,N,6]
            nbsum = nbm.sum(-1)
            nbcnt = nocc.sum(-1)
            lap = nbsum - nbcnt.unsqueeze(1) * m
            react = torch.tanh(torch.einsum("bij,bjn->bin", Wm, m) + bm.unsqueeze(-1) +
                               torch.cat([pol.unsqueeze(1), torch.zeros(B, k - 1, N, device=d)], 1))
            m = (m + 0.5 * (diff.unsqueeze(-1) * lap + react - decay.unsqueeze(-1) * m)) * occf.unsqueeze(1)
            cand = (~occ) & (nbcnt > 0)
            nbmean = nbsum / nbcnt.clamp(min=1).unsqueeze(1)
            gs = (gw.unsqueeze(-1) * nbmean).sum(1) - gthr.unsqueeze(-1) + dnoise.unsqueeze(-1) * self.hnoise(seeds, g, self.cell_idx)
            allowed = cand & (gs > 0)
            remaining = (maxn - occ.sum(1)).clamp(min=0)
            score = torch.where(allowed, gs, torch.full_like(gs, -1e9))
            order = torch.argsort(score, dim=1, descending=True)
            rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N, device=d).expand(B, N))
            chosen = allowed & (rank < remaining.unsqueeze(1))
            if not bool(chosen.any()):
                if record:
                    frames.append(occ.clone())
                continue
            first_nb = torch.argmax(nocc, dim=-1)                    # primeiro vizinho ocupado = célula-mãe
            par = torch.gather(self.nb_safe.unsqueeze(0).expand(B, N, 6), 2, first_nb.unsqueeze(-1)).squeeze(-1)
            parent = torch.where(chosen, par, parent)
            birth = torch.where(chosen, torch.full_like(birth, g + 1), birth)
            m = torch.where(chosen.unsqueeze(1), nbmean, m)
            occ = occ | chosen
            if record:
                frames.append(occ.clone())
        Wp, bp = L.view(G, "body_Wp"), L.view(G, "body_bp")
        props = torch.sigmoid(torch.einsum("bpk,bkn->bnp", Wp, m) + bp.unsqueeze(1)) * occ.unsqueeze(-1)
        innerv = torch.tanh(torch.einsum("bmk,bkn->bnm", L.view(G, "body_innerv"), m)) * occ.unsqueeze(-1)
        phase = math.pi * torch.tanh(m[:, -1] + m[:, 0] * 0.5)
        return Blueprint(occ=occ, props=props, morph=m, innerv=innerv, phase=phase, osc=L.view(G, "body_osc"),
                         parent_cell=parent, birth_step=birth, frames=frames)

    # ------------------------------------------------------------------ métricas de organização (instrumentos externos)
    def grid3(self, occ: torch.Tensor) -> torch.Tensor:
        return occ.view(occ.shape[0], self.D, self.D, self.D)

    def symmetry(self, occ: torch.Tensor) -> Dict[str, torch.Tensor]:
        g = self.grid3(occ).float()
        iou = lambda a, b: (a * b).sum((1, 2, 3)) / ((a + b).clamp(max=1).sum((1, 2, 3)) + 1e-9)
        return {"bilateral_x": iou(g, g.flip(1)), "bilateral_y": iou(g, g.flip(2)),
                "radial_z": iou(g, torch.rot90(g, 1, (1, 2)))}

    def canalization(self, G: torch.Tensor, seeds: torch.Tensor, n: int) -> torch.Tensor:
        """Mesmo genoma, ruídos de desenvolvimento diferentes → IoU médio entre corpos (robustez do desenvolvimento)."""
        occs = [self.develop(G, seeds * 7919 + i + 1).occ.float() for i in range(n)]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = occs[i], occs[j]
                vals.append((a * b).sum(1) / ((a + b).clamp(max=1).sum(1) + 1e-9))
        return torch.stack(vals).mean(0) if vals else torch.ones(G.shape[0], device=self.device)
