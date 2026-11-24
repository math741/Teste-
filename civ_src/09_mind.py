# %% [markdown]
# ## 09 — SIGNAL ENGINE + LEARNING ENGINE: substrato de sinais com plasticidade evoluível
# Não há "neurônios" impostos: há **unidades de sinal** genéricas com recepção (sensores físicos), transmissão
# (conexões recorrentes), excitabilidade (viés), atraso/integração (constante de tempo por unidade), oscilação
# (dinâmica recorrente), modulação (um canal escalar difuso calculado pela própria rede, com pesos do genoma),
# plasticidade local (regra ABCD do genoma: Δw = η·m·(A·pós·pré + B·pré + C·pós + D) − λ(w − w₀)) e poda/criação de
# conexões pela mutação estrutural. Nenhuma unidade tem rótulo. Se uma região passa a funcionar como sistema nervoso,
# memória ou estado afetivo, isso é **detectado de fora** (módulos 11 e 14).
# Separação explícita: GENÉTICO (w₀ e regras no genoma) · DESENVOLVIMENTO (corpo) · APRENDIZADO (w ≠ w₀ durante a vida)
# · CULTURA (informação adquirida socialmente, medida em 12).

# %%
@dataclass
class MindIntervention:
    """Intervenções de laboratório (só em arenas/forks marcados): nunca usadas no mundo principal sem registro."""
    sensor_mask: Optional[torch.Tensor] = None        # [n, I] multiplicador
    sensor_add: Optional[torch.Tensor] = None         # [n, I]
    clamp_dir: Optional[torch.Tensor] = None          # [n, N] direção no espaço de estados
    clamp_value: Optional[torch.Tensor] = None        # [n] deslocamento ao longo da direção
    freeze_plasticity: Optional[torch.Tensor] = None  # [n] bool
    zero_units: Optional[torch.Tensor] = None         # [n, N] bool (ablação de unidades)
    no_noise: bool = False


class LearningEngine:
    """Plasticidade local evoluível (regra ABCD do genoma) — aprendizado DURANTE a vida: w ≠ w₀, nunca altera o genoma."""

    def __init__(self, layout: GenomeLayout):
        self.L = layout

    def update(self, pool: AgentPool, hi: int, G: torch.Tensor, W: torch.Tensor, y: torch.Tensor, y_prev: torch.Tensor,
               mod: torch.Tensor, alive: torch.Tensor, iv: Optional[MindIntervention]) -> None:
        L = self.L
        eta = L.view(G, "plast_eta") * mod
        if iv is not None and iv.freeze_plasticity is not None:
            eta = torch.where(iv.freeze_plasticity, torch.zeros_like(eta), eta)
        abcd = L.view(G, "plast_abcd")
        A, Bc, C, D = abcd[:, 0], abcd[:, 1], abcd[:, 2], abcd[:, 3]
        hebb = (A.view(-1, 1, 1) * y.unsqueeze(2) * y_prev.unsqueeze(1) + Bc.view(-1, 1, 1) * y_prev.unsqueeze(1)
                + C.view(-1, 1, 1) * y.unsqueeze(2) + D.view(-1, 1, 1))
        W0 = L.view(G, "ctl_W0")
        dW = eta.view(-1, 1, 1) * hebb - L.view(G, "plast_decay").view(-1, 1, 1) * (W - W0)
        pool.W[:hi] = torch.where(alive.view(-1, 1, 1), (W + dW).clamp(-3, 3), W)


class SignalEngine:
    """Dinâmica do substrato de sinais (recepção, transmissão recorrente, integração com constante de tempo, oscilação,
    modulação difusa) + chamada ao LearningEngine. Saídas = canais de ação física."""

    def __init__(self, cfg: CivConfig, layout: GenomeLayout, device: torch.device):
        self.cfg, self.L, self.device = cfg, layout, device
        self.learning = LearningEngine(layout)

    def step(self, pool: AgentPool, hi: int, s: torch.Tensor, gen: torch.Generator,
             iv: Optional[MindIntervention] = None) -> torch.Tensor:
        L = self.L
        G = pool.genome[:hi]
        alive = pool.alive[:hi]
        if iv is not None and iv.sensor_mask is not None:
            s = s * iv.sensor_mask
        if iv is not None and iv.sensor_add is not None:
            s = s + iv.sensor_add
        Win, Wout = L.view(G, "ctl_Win"), L.view(G, "ctl_Wout")
        b, bout = L.view(G, "ctl_b"), L.view(G, "ctl_bout")
        tau = torch.exp(L.view(G, "ctl_logtau"))
        x, W = pool.x[:hi], pool.W[:hi]
        y_prev = torch.tanh(x)
        pre = torch.bmm(Win, s.unsqueeze(-1)).squeeze(-1) + torch.bmm(W, y_prev.unsqueeze(-1)).squeeze(-1) + b
        x_new = x + (-x + pre) / tau
        if not (iv is not None and iv.no_noise):
            # exploração motora (motor babbling): ruído interno que decai com a idade, com cronograma do genoma;
            # gerador semeado → reprodutível em SCIENTIFIC_REPLAY
            sig = L.view(G, "babble_sigma") * torch.exp(-pool.age[:hi] / L.view(G, "babble_tau"))
            x_new = x_new + sig.unsqueeze(1) * torch.randn(x_new.shape, generator=gen, device=self.device)
        if iv is not None and iv.clamp_dir is not None:
            proj = (x_new * iv.clamp_dir).sum(1, keepdim=True)
            x_new = x_new + (iv.clamp_value.unsqueeze(1) - proj) * iv.clamp_dir
        if iv is not None and iv.zero_units is not None:
            x_new = torch.where(iv.zero_units, torch.zeros_like(x_new), x_new)
        x_new = x_new.clamp(-6, 6)
        y = torch.tanh(x_new)
        out = torch.bmm(Wout, y.unsqueeze(-1)).squeeze(-1) + bout
        mod = torch.sigmoid((L.view(G, "ctl_wmod") * y).sum(1) + L.view(G, "ctl_bmod"))
        self.learning.update(pool, hi, G, W, y, y_prev, mod, alive, iv)
        pool.x[:hi] = torch.where(alive.view(-1, 1), x_new, x)
        pool.mod[:hi] = torch.where(alive, mod, pool.mod[:hi])
        out = torch.where(alive.view(-1, 1), out, torch.zeros_like(out))
        pool.out[:hi] = out
        return out


MindEngine = SignalEngine      # nome histórico usado pelos demais módulos (substrato de sinais + aprendizado)
