# %% [markdown]
# ## 06 — MATTER ENGINE: matéria manipulável do ambiente
# Não existe `build_house()`. Existe matéria: depósitos naturais por tile, matéria carregada pelos agentes e pilhas
# construídas (voxels esparsos sobre o mapa). Ações físicas genéricas: **pegar** (do depósito ou do topo de uma pilha à
# frente) e **depositar** (no tile à frente: empilha; sobre água, aterra → tile vira caminhável). Pilhas ≥2 bloqueiam
# passagem; tiles cercados por paredes ficam abrigados (reduz exposição — vantagem funcional, não rótulo).
# Matéria é conservada: desmoronamento por intempérie devolve o bloco ao depósito do tile (teste de conservação no boot).

# %%
def rank_within_groups(keys: torch.Tensor) -> torch.Tensor:
    """Posição de cada elemento dentro do seu grupo (mesma chave), em ordem estável. Vetorizado."""
    if keys.numel() == 0:
        return keys
    order = torch.argsort(keys, stable=True)
    sk = keys[order]
    first = torch.ones_like(sk, dtype=torch.bool)
    first[1:] = sk[1:] != sk[:-1]
    idx = torch.arange(sk.numel(), device=keys.device)
    start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
    rank_sorted = idx - start
    rank = torch.empty_like(rank_sorted)
    rank[order] = rank_sorted
    return rank


class StructureLedger:
    """Registro de blocos (quem construiu, quando, onde, nível) para história, visualização e testes. Capacidade fixa."""

    def __init__(self, cap: int, device: torch.device):
        self.cap, self.device = int(cap), device
        self.tile = torch.full((self.cap,), -1, dtype=torch.int64, device=device)
        self.z = torch.zeros(self.cap, dtype=torch.int32, device=device)
        self.owner = torch.zeros(self.cap, dtype=torch.int64, device=device)
        self.tick = torch.zeros(self.cap, dtype=torch.int64, device=device)
        self.kind = torch.zeros(self.cap, dtype=torch.int8, device=device)       # 0 pilha, 1 aterro
        self.active = torch.zeros(self.cap, dtype=torch.bool, device=device)
        self.ptr = 0
        self.total_placed = 0
        self.total_removed = 0

    def add(self, tiles: torch.Tensor, z: torch.Tensor, owners: torch.Tensor, tick: int, kind: int):
        n = tiles.numel()
        if n == 0:
            return
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.tile[idx] = tiles; self.z[idx] = z.int(); self.owner[idx] = owners; self.tick[idx] = tick
        self.kind[idx] = kind; self.active[idx] = True
        self.ptr = (self.ptr + n) % self.cap
        self.total_placed += n

    def remove_top(self, tiles: torch.Tensor, z: torch.Tensor):
        if tiles.numel() == 0:
            return
        hit = self.active & torch.isin(self.tile, tiles)
        if bool(hit.any()):
            cand = torch.nonzero(hit).flatten()
            key = self.tile[cand] * 64 + self.z[cand].long()
            want = tiles * 64 + z.long()
            sel = cand[torch.isin(key, want)]
            self.active[sel] = False
            self.total_removed += int(sel.numel())

    def state_dict(self):
        idx = torch.nonzero(self.active).flatten()
        return {"cap": self.cap, "ptr": self.ptr, "placed": self.total_placed, "removed": self.total_removed,
                **{k: getattr(self, k)[idx].cpu().numpy() for k in ["tile", "z", "owner", "tick", "kind"]}}

    def load_state_dict(self, s):
        n = len(s["tile"])
        n = min(n, self.cap)
        for k in ["tile", "z", "owner", "tick", "kind"]:
            getattr(self, k)[:n] = torch.as_tensor(s[k][:n], device=self.device)
        self.active[:] = False; self.active[:n] = True
        self.ptr = n % self.cap; self.total_placed = int(s["placed"]); self.total_removed = int(s["removed"])


class MatterEngine:
    def __init__(self, cfg: CivConfig, device: torch.device):
        self.cfg, self.device = cfg, device

    def ahead_tile(self, ws: WorldState, pos: torch.Tensor, heading: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tx = (pos[:, 0] + torch.cos(heading) * 1.0).floor().long().clamp(0, ws.W - 1)
        ty = (pos[:, 1] + torch.sin(heading) * 1.0).floor().long().clamp(0, ws.H - 1)
        return ty, tx

    def pick(self, ws: WorldState, ledger: StructureLedger, idx: torch.Tensor, pos: torch.Tensor, heading: torch.Tensor,
             carried: torch.Tensor) -> torch.Tensor:
        """Retorna máscara de sucesso (por agente em idx). Do depósito do próprio tile; se vazio, desmonta o topo à frente."""
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        iy, ix = WorldEngine.tile_index(ws, pos)
        tile = iy * ws.W + ix
        cap_ok = carried < self.cfg.carry_capacity
        rank = rank_within_groups(tile)
        dep = ws.deposit.flatten()[tile]
        from_dep = cap_ok & (rank.float() < dep)
        cnt = torch.zeros(ws.H * ws.W, device=self.device).index_add_(0, tile[from_dep], torch.ones(int(from_dep.sum()), device=self.device))
        ws.deposit = (ws.deposit.flatten() - cnt).clamp(min=0).view(ws.H, ws.W)
        ty, tx = self.ahead_tile(ws, pos, heading)
        atile = ty * ws.W + tx
        want = cap_ok & ~from_dep & (ws.struct_h.flatten()[atile] > 0) & (ws.struct_integrity.flatten()[atile] < 1e8)
        ok_struct = torch.zeros_like(want)
        if bool(want.any()):
            sel = torch.nonzero(want).flatten()
            r2 = rank_within_groups(atile[sel])
            sel = sel[r2 == 0]                                   # um desmonte por tile por tick
            ok_struct[sel] = True
            t = atile[sel]
            hz = ws.struct_h.flatten()[t]
            ledger.remove_top(t, hz)
            sh = ws.struct_h.flatten().clone(); sh[t] = hz - 1
            ws.struct_h = sh.view(ws.H, ws.W)
            ws.struct_dirty = True
        return from_dep | ok_struct

    def place(self, ws: WorldState, ledger: StructureLedger, pos: torch.Tensor, heading: torch.Tensor,
              owners: torch.Tensor, tick: int) -> torch.Tensor:
        n = pos.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        ty, tx = self.ahead_tile(ws, pos, heading)
        t = ty * ws.W + tx
        water = ws.is_water.flatten()[t] & ~ws.filled.flatten()[t]
        frozen_wall = ws.struct_integrity.flatten()[t] >= 1e8
        rank = rank_within_groups(t)
        h = ws.struct_h.flatten()[t]
        ok_fill = water & (rank == 0)
        ok_stack = (~water) & ~frozen_wall & (h + rank < self.cfg.max_stack)
        if bool(ok_fill.any()):
            f = ws.filled.flatten().clone(); f[t[ok_fill]] = True
            ws.filled = f.view(ws.H, ws.W)
            ledger.add(t[ok_fill], torch.zeros(int(ok_fill.sum()), device=self.device), owners[ok_fill], tick, 1)
        if bool(ok_stack.any()):
            ts = t[ok_stack]
            add = torch.zeros(ws.H * ws.W, dtype=torch.int32, device=self.device).index_add_(
                0, ts, torch.ones(ts.numel(), dtype=torch.int32, device=self.device))
            levels = h[ok_stack] + rank[ok_stack].int() + 1
            ws.struct_h = (ws.struct_h.flatten() + add).view(ws.H, ws.W)
            si = ws.struct_integrity.flatten().clone(); si[ts] = torch.clamp(si[ts], min=1.0)
            ws.struct_integrity = si.view(ws.H, ws.W)
            ledger.add(ts, levels, owners[ok_stack], tick, 0)
            ws.struct_dirty = True
        return ok_fill | ok_stack

    def total_material(self, ws: WorldState, carried: torch.Tensor) -> float:
        built = ws.struct_h.clamp(min=0) * (ws.struct_integrity < 1e8)
        return float(ws.deposit.sum() + carried.sum() + built.sum() + ws.filled.sum())
