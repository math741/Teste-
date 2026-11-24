# %% [markdown]
# ## 17 — TELEMETRY ENGINE: observador puro (protocolo versionado, deltas binários, fila limitada)
# A telemetria **nunca** entra no caminho crítico da ciência: roda na thread da simulação entre ticks, só lê estado,
# não consome o RNG do mundo e não altera fidelidade científica. Publica em frequência controlada (`telemetry_hz`):
# * `base` — relevo/água/biomas reduzidos + estruturas (quando muda muito ou sob pedido);
# * `frame` — agentes (subconjunto estável: corpos F2+, selecionado, região da câmera, amostra por hash do ID),
#   posições quantizadas uint16, rumo, cor genética, fidelidade, energia, tipo cultural; deltas de estruturas;
#   eventos novos; luz/tempestade;
# * `body` — nós/sinais reais do corpo (F2+) ou *gêmeo visual* (desenvolvido do genoma; marcado como visual);
# * `status`, `lab` (câmera do cientista), `heat` (camadas), `ring` (replay temporal do mundo).
# Pedidos REST que precisam de tensores (inspetor de agente etc.) entram numa fila atendida pela simulação.

# %%
def ensure_package(module: str, pip_spec: str) -> bool:
    """Instala só o necessário (versões com limites compatíveis) quando ausente."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_spec], timeout=300)
            importlib.invalidate_caches()
            importlib.import_module(module)
            return True
        except Exception as e:
            print(f"[deps] não foi possível instalar {pip_spec}: {e}")
            return False


HAVE_MSGPACK = ensure_package("msgpack", "msgpack>=1.0,<2")
if HAVE_MSGPACK:
    import msgpack

LIVE_PROTOCOL = {"name": "genesis-live", "version": RENDER_PROTOCOL_VERSION, "telemetry": TELEMETRY_SCHEMA_VERSION,
                 "events": EVENT_SCHEMA_VERSION}
HEAT_LAYERS = ["population", "resources", "activity", "species", "culture", "communication", "settlements", "novelty",
               "temperature", "storm"]


class ObserverDesk:
    """Ponte thread-safe servidor ↔ simulação (sem acesso direto a tensores a partir do servidor)."""

    def __init__(self, maxlen: int = 4):
        self.lock = threading.Lock()
        self.frames: "collections.deque" = collections.deque(maxlen=maxlen)
        self.status: Dict[str, Any] = {}
        self.base: Optional[bytes] = None
        self.base_seq = 0
        self.body: Dict[int, bytes] = {}
        self.lab: Optional[bytes] = None
        self.heat: Dict[str, bytes] = {}
        self.ring_index: List[Dict[str, Any]] = []
        self.requests: "queue.Queue" = queue.Queue(maxsize=256)
        self.selected: Dict[str, int] = {}                 # cliente → aid selecionado (visual)
        self.layers: Dict[str, str] = {}
        self.views: Dict[str, Tuple[float, float, float, float, int]] = {}
        self.want_base = False
        self.admin_queue: "queue.Queue" = queue.Queue(maxsize=32)
        self.seq = 0
        self.online = True

    def publish_frame(self, payload: bytes):
        with self.lock:
            self.seq += 1
            self.frames.append((self.seq, payload))

    def latest(self) -> Optional[Tuple[int, bytes]]:
        with self.lock:
            return self.frames[-1] if self.frames else None

    def ask(self, kind: str, args: Dict[str, Any], timeout: float = 4.0) -> Any:
        """Chamado pelo servidor: enfileira pedido e espera a simulação responder (com timeout)."""
        ev = threading.Event()
        box: Dict[str, Any] = {}
        try:
            self.requests.put_nowait((kind, args, ev, box))
        except queue.Full:
            return {"error": "busy"}
        if not ev.wait(timeout):
            return {"error": "timeout", "note": "simulação ocupada; tente novamente"}
        return box.get("result")


def pack_msg(obj: Dict[str, Any]) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


class TelemetryEngine:
    def __init__(self, uni: "UniverseEngine", desk: ObserverDesk):
        self.u, self.desk = uni, desk
        self.cfg = uni.cfg
        self.last_frame = 0.0
        self.last_status = 0.0
        self.last_heat = 0.0
        self.last_body = 0.0
        self.last_lab = 0.0
        self.last_struct: Optional[torch.Tensor] = None
        self.last_filled: Optional[torch.Tensor] = None
        self.events_sent = 0
        self.ring: "collections.deque" = collections.deque(maxlen=uni.cfg.snapshot_ring)
        self.ring_last_tick = -10 ** 9
        self.tps = 0.0
        self._tps_t, self._tps_tick = time.time(), uni.tick_count
        self.visual = BodyRegistry(uni.cfg, uni.dev, uni.body, 4, uni.cfg.body_substeps)
        self.visual_aid: Dict[int, int] = {}
        self.xhist: Dict[int, "collections.deque"] = {}
        self.frames_sent = 0
        self.bytes_sent = 0
        self.state_label = "BOOTING"
        self.checkpoint_info: Dict[str, Any] = {}
        self.session_start = time.time()
        self.lab_view_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None

    # ------------------------------------------------------------------ base (relevo, água, biomas, estruturas)
    def build_base(self) -> bytes:
        ws = self.u.ws
        step = max(1, math.ceil(max(ws.H, ws.W) / 256))
        h = ws.height[::step, ::step]
        wat = (ws.is_water & ~ws.filled)[::step, ::step]
        temp = ws.temp_base[::step, ::step]; moist = ws.moisture[::step, ::step]; fert = ws.fertility[::step, ::step]
        hn = (h - h.min()) / (h.max() - h.min() + 1e-6)
        # cor de bioma = função declarada de (altura, temperatura, umidade, fertilidade); legenda no viewer
        r = (0.35 + 0.4 * hn - 0.25 * fert).clamp(0, 1); g = (0.3 + 0.5 * fert + 0.1 * moist).clamp(0, 1)
        b = (0.2 + 0.2 * moist + 0.35 * (temp < 0.25).float()).clamp(0, 1)
        snow = (temp < 0.18) & ~wat
        r = torch.where(snow, torch.full_like(r, 0.92), r); g = torch.where(snow, torch.full_like(g, 0.94), g)
        b = torch.where(snow, torch.full_like(b, 0.97), b)
        rgb = (torch.stack([r, g, b], -1) * 255).to(torch.uint8)
        hq = ((h / max(1e-6, self.cfg.height_scale)).clamp(0, 1) * 65535).to(torch.int32).to(torch.int16)
        st = self._structures()
        self.last_struct = ws.struct_h.clone(); self.last_filled = ws.filled.clone()
        msg = {"t": "base", "proto": LIVE_PROTOCOL, "tick": self.u.tick_count, "W": ws.W, "H": ws.H, "step": step,
               "gw": int(h.shape[1]), "gh": int(h.shape[0]), "height_scale": self.cfg.height_scale,
               "sea_level": ws.flags.get("sea_level", 0.0), "height": hq.cpu().numpy().tobytes(),
               "water": wat.cpu().numpy().astype(np.uint8).tobytes(), "rgb": rgb.cpu().numpy().tobytes(),
               "struct": st.tobytes(), "struct_n": int(st.shape[0]), "max_stack": self.cfg.max_stack,
               "settlements": self._settlements(), "roads": self._roads(),
               "legend": {"terrain": "cor = f(altura, temperatura, umidade, fertilidade); neve = temperatura < 0.18",
                          "agents": "cor = assinatura visual genética (display) — identidade, não semântica",
                          "blocks": "matéria construída real (pilhas); azul-esverdeado = aterro sobre água"}}
        return pack_msg(msg)

    def _structures(self) -> np.ndarray:
        ws = self.u.ws
        built = (ws.struct_h > 0) & (ws.struct_integrity < 1e8)
        ys, xs = torch.nonzero(built, as_tuple=True)
        hs = ws.struct_h[ys, xs]
        fy, fx = torch.nonzero(ws.filled, as_tuple=True)
        a = torch.stack([xs, ys, hs.long(), torch.zeros_like(xs)], 1)
        b = torch.stack([fx, fy, torch.ones_like(fx), torch.ones_like(fx)], 1)
        out = torch.cat([a, b])[:200000]
        return out.to(torch.int16).cpu().numpy()

    def _settlements(self) -> List[Dict[str, Any]]:
        out = []
        for sid, s in self.u.settlements.settlements.items():
            if s["state"] in ("detected", "city", "collapsed", "candidate"):
                out.append({"id": sid, "state": s["state"], "centre": s.get("centre", [0, 0]), "pop": s.get("pop", 0),
                            "blocks": s.get("blocks", 0), "cells": len(s.get("cells", [])), "cell": self.u.settlements.cell,
                            "founded": s.get("founded")})
        return out

    def _roads(self) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "tiles": r["tiles"], "bbox": r["bbox"], "settlements": r["settlements"]}
                for r in self.u.settlements.road_segments[:100]]

    # ------------------------------------------------------------------ frame dinâmico
    def _select_agents(self, max_n: int) -> torch.Tensor:
        pool = self.u.pool
        alive = torch.nonzero(pool.alive).flatten()
        if alive.numel() <= max_n:
            return alive
        pri = torch.zeros(alive.numel(), device=pool.device)
        pri += (pool.fidelity[alive] >= 2).float() * 10
        sel = set(self.desk.selected.values())
        if sel:
            pri += torch.isin(pool.aid[alive], torch.tensor(sorted(sel), device=pool.device)).float() * 100
        views = list(self.desk.views.values())
        if views:
            x0, y0, x1, y1, _ = views[-1]
            p = pool.pos[alive]
            inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
            pri += inside.float() * 5
        # amostra estável (hash do ID) → o mesmo agente aparece em frames seguidos (interpolação suave)
        h = ((pool.aid[alive] * 2654435761) % 1000).float() / 1000
        pri += h
        return alive[torch.argsort(pri, descending=True)[:max_n]]

    def build_frame(self) -> bytes:
        u, pool, ws = self.u, self.u.pool, self.u.ws
        max_n = self.cfg.max_stream_agents
        if self.desk.views:
            max_n = min(max_n, max(50, list(self.desk.views.values())[-1][4]))
        idx = self._select_agents(max_n)
        pos = pool.pos[idx]
        q = torch.stack([(pos[:, 0] / ws.W * 65535), (pos[:, 1] / ws.H * 65535)], 1).clamp(0, 65535).to(torch.int32)
        msg: Dict[str, Any] = {"t": "frame", "tick": u.tick_count, "ts": time.time(), "n": int(idx.numel()),
                               "ids": pool.aid[idx].to(torch.int64).cpu().numpy().tobytes(),
                               "pos": q.to(torch.int32).cpu().numpy().astype(np.uint16).tobytes(),
                               "head": ((pool.heading[idx] / (2 * math.pi)) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "color": (pool.color[idx] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "fid": pool.fidelity[idx].to(torch.uint8).cpu().numpy().tobytes(),
                               "energy": (pool.energy[idx] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "culture": (pool.culture_type[idx] + 1).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "carry": pool.carried[idx].clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "emit": (pool.emit[idx].max(1).values * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                               "light_phase": float((u.tick_count % self.cfg.day_period) / self.cfg.day_period),
                               "season": float(math.sin(2 * math.pi * u.tick_count / max(1, self.cfg.season_period)))}
        # deltas de estruturas
        if self.last_struct is not None and self.last_struct.shape == ws.struct_h.shape:
            ch = (ws.struct_h != self.last_struct) | (ws.filled != self.last_filled)
            if bool(ch.any()):
                ys, xs = torch.nonzero(ch, as_tuple=True)
                if ys.numel() > 20000:
                    self.desk.want_base = True
                else:
                    hv = torch.where(ws.struct_integrity[ys, xs] < 1e8, ws.struct_h[ys, xs], torch.zeros_like(ws.struct_h[ys, xs]))
                    d = torch.stack([xs, ys, hv.long(), ws.filled[ys, xs].long()], 1).to(torch.int16)
                    msg["sdelta"] = d.cpu().numpy().tobytes()
                    msg["sdelta_n"] = int(d.shape[0])
                self.last_struct = ws.struct_h.clone(); self.last_filled = ws.filled.clone()
        storm = F.adaptive_avg_pool2d(ws.storm[None, None], (32, 32))[0, 0]
        msg["storm"] = (storm * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes()
        evs = list(u.events_out)
        msg["events"] = to_jsonable(evs[self.events_sent:] if self.events_sent <= len(evs) else evs[-20:])[-30:]
        self.events_sent = len(evs)
        return pack_msg(msg)

    # ------------------------------------------------------------------ corpo (real F2+ ou gêmeo visual)
    def build_body(self, aid: int) -> Optional[bytes]:
        pool = self.u.pool
        hit = torch.nonzero(pool.alive & (pool.aid == aid)).flatten()
        if hit.numel() == 0:
            return None
        s = int(hit[0])
        fid = int(pool.fidelity[s])
        reg, bs, visual = None, None, False
        for lvl, r in self.u.bodies.items():
            m = torch.nonzero(r.agent_slot == s).flatten()
            if m.numel():
                reg, bs = r, int(m[0]); break
        if reg is None:
            visual = True
            if aid not in self.visual_aid:
                self.visual.detach(torch.nonzero(self.visual.agent_slot >= 0).flatten())
                self.visual_aid = {}
                got = self.visual.attach(torch.tensor([s], device=pool.device), pool.aid[s:s + 1], pool.genome[s:s + 1],
                                         pool.aid[s:s + 1] % 100003)
                if got.numel() == 0:
                    return None
                self.visual_aid[aid] = int(got[0])
            reg, bs = self.visual, self.visual_aid[aid]
            motor = torch.zeros(reg.cap, 2, device=pool.device)
            motor[bs, 0] = torch.tanh(pool.out[s, 0]); motor[bs, 1] = torch.tanh(pool.out[s, 1])
            reg.body.step(reg.bb, motor)                              # física do gêmeo visual (não afeta a ciência)
        bb = reg.bb
        act = bb.active[bs]
        P = bb.pos[bs]
        E = reg.dev.edges
        on = bb.e_on[bs]
        props = bb.props[bs]
        x = pool.x[s]
        dq = self.xhist.setdefault(aid, collections.deque(maxlen=120))
        dq.append(x.detach().cpu().numpy().astype(np.float16))
        msg = {"t": "body", "aid": aid, "fidelity": fid, "visual_twin": visual, "N": int(P.shape[0]),
               "nodes": P.detach().cpu().numpy().astype(np.float32).tobytes(), "active": act.cpu().numpy().astype(np.uint8).tobytes(),
               "signal": (bb.signal[bs].clamp(0, 4) / 4 * 255).to(torch.uint8).cpu().numpy().tobytes(),
               "edges": E[on].to(torch.int16).cpu().numpy().tobytes(), "n_edges": int(on.sum()),
               "props": (props.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().tobytes(), "prop_names": PROP_NAMES,
               "state": np.stack(list(dq)).tobytes(), "state_T": len(dq), "state_N": int(x.numel()),
               "note": ("gêmeo visual: corpo desenvolvido do genoma só para visualização; a ciência deste agente usa o resumo F1"
                        if visual else "corpo científico real (Matter Nodes)")}
        return pack_msg(msg)

    # ------------------------------------------------------------------ camadas (heatmaps)
    def build_heat(self, layer: str, R: int = 64) -> bytes:
        u, ws, pool = self.u, self.u.ws, self.u.pool
        dev = ws.device

        def pool2(x):
            return F.adaptive_avg_pool2d(x[None, None].float(), (R, R))[0, 0]
        alive = pool.alive
        p = pool.pos[alive]
        cy = (p[:, 1] / ws.H * R).long().clamp(0, R - 1); cx = (p[:, 0] / ws.W * R).long().clamp(0, R - 1)
        cnt = torch.bincount(cy * R + cx, minlength=R * R).float().view(R, R)
        if layer == "population":
            g = cnt
        elif layer == "resources":
            g = pool2(ws.resource)
        elif layer == "activity":
            g = pool2(ws.traffic)
        elif layer == "communication":
            g = pool2(ws.sound.sum(0))
        elif layer == "temperature":
            g = pool2(ws.temperature)
        elif layer == "storm":
            g = pool2(ws.storm)
        elif layer == "settlements":
            g = pool2(((ws.struct_h > 0) & (ws.struct_integrity < 1e8)).float() + ws.filled.float())
        elif layer in ("species", "culture", "novelty"):
            if layer == "species":
                val = (pool.genome[alive][:, : min(8, self.u.L.P)].sum(1) * 7.31) % 1.0
            elif layer == "culture":
                val = (pool.culture_type[alive].float() + 1) / 8
            else:
                sig = pool.signature[alive]
                val = (sig - sig.mean(0)).norm(dim=-1) if sig.shape[0] else sig.sum(1)
            s = torch.zeros(R * R, device=dev).index_add_(0, cy * R + cx, val.float())
            g = (s / cnt.flatten().clamp(min=1)).view(R, R)
        else:
            g = cnt
        g = g / g.max().clamp(min=1e-9)
        return pack_msg({"t": "heat", "layer": layer, "R": R, "tick": u.tick_count,
                         "data": (g * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes(),
                         "note": "camada derivada do estado real; cores por camada não implicam semântica"})

    # ------------------------------------------------------------------ status
    def build_status(self) -> Dict[str, Any]:
        u = self.u
        now = time.time()
        if now - self._tps_t >= 1.0:
            self.tps = (u.tick_count - self._tps_tick) / (now - self._tps_t)
            self._tps_t, self._tps_tick = now, u.tick_count
        gpu = {}
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                gpu = {"name": torch.cuda.get_device_name(0), "allocated": int(torch.cuda.memory_allocated()),
                       "reserved": int(torch.cuda.memory_reserved()), "peak": int(torch.cuda.max_memory_allocated()),
                       "free": int(free), "total": int(total)}
                try:
                    gpu["util"] = int(torch.cuda.utilization())
                except Exception:
                    pass
            except Exception as e:
                gpu = {"error": repr(e)}
        st = u.status()
        sci = u.scientist.status()
        maxgen = int(u.pool.generation[u.pool.alive].max()) if u.pool.n_alive() else 0
        return {"t": "status", "state": self.state_label, "tick": u.tick_count, "tps": round(self.tps, 1),
                "speed_label": f"{self.tps:.0f} ticks/s (1 tick = unidade artificial)", "population": st["population"],
                "f0_population": st["f0_population"], "bodies": st["bodies"], "generation": maxgen,
                "qd_generations": st["qd_generations"], "h_level": st["h_level"], "a_level": st["a_level"],
                "c_level": st["civ_level"], "c_name": st["civ_level_name"],
                "h_name": H_LEVELS[st["h_level"]] if st["h_level"] >= 0 else "—",
                "a_name": A_LEVELS[st["a_level"]] if st["a_level"] >= 0 else "—",
                "discoveries": st["discoveries"], "settlements": st["settlements"], "science": sci,
                "checkpoint": self.checkpoint_info, "gpu": gpu, "session_runtime": now - self.session_start,
                "total_runtime": u.total_runtime + (now - self.session_start), "paused": u.paused,
                "profiler": dict(list(u.prof.report().items())[:10]), "world": {"W": u.ws.W, "H": u.ws.H},
                "day_phase": float((u.tick_count % self.cfg.day_period) / self.cfg.day_period),
                "time_labels": {"tick": "unidade de tempo artificial", "day": f"{self.cfg.day_period} ticks (artificial)",
                                "season": f"{self.cfg.season_period} ticks (artificial)"},
                "frames_sent": self.frames_sent, "proto": LIVE_PROTOCOL, "version": GENESIS_VERSION,
                "sentinel": u.sentinel.last, "ethical_triggers": len(u.sentinel.triggers)}

    # ------------------------------------------------------------------ ciclo (chamado pela simulação)
    def pump(self, force: bool = False):
        now = time.time()
        d = self.desk
        try:
            if d.base is None or d.want_base:
                d.base = self.build_base(); d.base_seq += 1; d.want_base = False
            if force or now - self.last_frame >= 1.0 / max(0.5, self.cfg.telemetry_hz):
                d.publish_frame(self.build_frame()); self.last_frame = now; self.frames_sent += 1
            if now - self.last_status >= 1.0:
                st = self.build_status()
                with d.lock:
                    d.status = st
                self.last_status = now
            if d.selected and now - self.last_body >= 0.25:
                for aid in set(d.selected.values()):
                    b = self.build_body(int(aid))
                    if b is not None:
                        d.body[int(aid)] = b
                self.last_body = now
            if now - self.last_heat >= 2.0:
                for layer in set(d.layers.values()):
                    if layer in HEAT_LAYERS:
                        d.heat[layer] = self.build_heat(layer)
                self.last_heat = now
            if self.lab_view_fn is not None and now - self.last_lab >= 0.5:
                v = self.lab_view_fn()
                d.lab = pack_msg({"t": "lab", **{k: (val.tobytes() if isinstance(val, np.ndarray) else val)
                                                  for k, val in v.items()}, "pos_n": int(len(v["pos"]))}) if v else None
                self.last_lab = now
            if self.u.tick_count - self.ring_last_tick >= self.cfg.snapshot_every:
                self._ring_push()
        except Exception as e:
            if self.u.logger:
                self.u.logger.event("telemetry_error", error=repr(e), trace=traceback.format_exc()[-1500:])
        self.service(budget_s=0.02)

    def _ring_push(self):
        pool, ws = self.u.pool, self.u.ws
        idx = self._select_agents(min(2000, self.cfg.max_stream_agents))
        pos = pool.pos[idx]
        q = torch.stack([(pos[:, 0] / ws.W * 65535), (pos[:, 1] / ws.H * 65535)], 1).clamp(0, 65535).to(torch.int32)
        self.ring.append({"tick": self.u.tick_count, "ids": pool.aid[idx].cpu().numpy().astype(np.int64),
                          "pos": q.cpu().numpy().astype(np.uint16), "color": (pool.color[idx] * 255).to(torch.uint8).cpu().numpy(),
                          "pop": self.u.pool.n_alive()})
        self.ring_last_tick = self.u.tick_count
        with self.desk.lock:
            self.desk.ring_index = [{"i": i, "tick": r["tick"], "pop": r["pop"]} for i, r in enumerate(self.ring)]

    def ring_frame(self, i: int) -> Optional[bytes]:
        if not 0 <= i < len(self.ring):
            return None
        r = self.ring[i]
        return pack_msg({"t": "ring", "i": i, "tick": r["tick"], "n": int(len(r["ids"])), "ids": r["ids"].tobytes(),
                         "pos": r["pos"].tobytes(), "color": r["color"].tobytes()})

    # ------------------------------------------------------------------ pedidos do servidor (inspetor etc.)
    def service(self, budget_s: float = 0.02):
        t0 = time.time()
        while time.time() - t0 < budget_s:
            try:
                kind, args, ev, box = self.desk.requests.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = self._answer(kind, args)
            except Exception as e:
                box["result"] = {"error": type(e).__name__, "detail": str(e)[:200]}
            ev.set()

    def _answer(self, kind: str, a: Dict[str, Any]) -> Any:
        u = self.u
        if kind == "agent":
            return agent_card(u, int(a["aid"]))
        if kind == "city":
            return u.settlements.public(int(a["sid"]))
        if kind == "ring":
            return self.ring_frame(int(a["i"]))
        if kind == "develop":
            return development_replay(u, int(a["aid"]))
        if kind == "search":
            return search_world(u, str(a["q"]))
        if kind == "history":
            return {"events": u.history.query(int(a.get("since", 0)), limit=int(a.get("limit", 300))),
                    "compressed": dict(u.history.compressed), "total": u.history.total,
                    "demography": u.demography.last(200)}
        if kind == "discoveries":
            ar = u.archive
            return {"discoveries": ar.discoveries[-300:], "pantheon": ar.pantheon, "overall": ar.overall_champion,
                    "convergence": ar.convergence[-20:], "anomalies": list(ar.weird)[-30:],
                    "failures": len(ar.failures), "ethical": u.sentinel.triggers[-20:], "forks": list(u.forks.keys())[-20:]}
        if kind == "ladders":
            return ladder_status(u)
        if kind == "lineage":
            aid = int(a["aid"])
            pool = u.pool
            anc = u.lineage.ancestors(aid, 30)
            kids = pool.aid[pool.alive & (pool.parent == aid)].cpu().numpy().tolist()[:200]
            alive_set = set(pool.aid[pool.alive].cpu().numpy().tolist()) if anc else set()
            sib = []
            p0 = u.lineage.parent_of(aid)
            if p0:
                sib = pool.aid[pool.alive & (pool.parent == p0)].cpu().numpy().tolist()[:50]
            return {"aid": aid, "ancestors": [{"aid": x, "alive": x in alive_set,
                                               "children": u.lineage.children_count.get(x, 0)} for x in anc],
                    "children_alive": kids, "siblings_alive": sib, "children_total": u.lineage.children_count.get(aid, 0)}
        if kind == "world":
            ws = u.ws
            return {"W": ws.W, "H": ws.H, "tick": u.tick_count, "sea_level": ws.flags.get("sea_level"), "seed": ws.seed,
                    "settlements": self._settlements(), "roads": self._roads(), "civ": to_jsonable(u.last.get("civ", {})),
                    "social": to_jsonable({k: v for k, v in u.last.get("social", {}).items() if k != "dissolved"}),
                    "culture": to_jsonable(u.last.get("culture", {})), "communication": to_jsonable(u.last.get("comm", {})),
                    "demography": to_jsonable(u.demography.last(60)), "fidelity": dict(u.fidelity_log),
                    "qd": to_jsonable(u.qd.summary()), "forks": to_jsonable({k: {kk: vv for kk, vv in v.items()
                                                                                    if kk not in ("divergence_map",)}
                                                                               for k, v in u.forks.items()})}
        raise ValueError("pedido desconhecido")


def agent_card(u: "UniverseEngine", aid: int) -> Dict[str, Any]:
    """Inspetor de agente: identidade persistente, corpo, idade, linhagem, recursos, aprendizado, memória, afeto
    funcional (só se houver eixo validado), relações, cultura, evidências."""
    pool = u.pool
    hit = torch.nonzero(pool.alive & (pool.aid == aid)).flatten()
    if hit.numel() == 0:
        anc = u.lineage.ancestors(aid, 20)
        return {"aid": aid, "alive": False, "ancestors": anc, "note": "agente não está vivo (ou foi agregado em F0)"}
    s = int(hit[0])
    G = pool.genome[s]
    gh = genome_hash(G.view(1, -1))
    W0 = u.L.view(G.view(1, -1), "ctl_W0")[0]
    learned = float((pool.W[s] - W0).norm() / (W0.norm() + 1e-6))
    cap = pool.body_cap[s].cpu().numpy()
    partners = u.social.neighbors(aid, u.tick_count, 8)
    ppos = []
    for o, w in partners:
        h2 = torch.nonzero(pool.alive & (pool.aid == o)).flatten()
        ppos.append({"aid": o, "w": round(w, 4), "pos": pool.pos[int(h2[0])].cpu().numpy().tolist() if h2.numel() else None})
    affect = {"status": "no validated functional-affect axis for this lineage", "claim": "NOT YET VALIDATED"}
    for wgh, w in u.sentinel.watch.items():
        if float(u.L.distance(G.view(1, -1), torch.as_tensor(w["genome"], device=G.device).view(1, -1))) < 0.08:
            proj = float(pool.x[s] @ torch.as_tensor(w["v"], device=G.device))
            affect = {"status": "validated A1 valence axis (lineage)", "projection": proj, "threshold": w["thr"],
                      "claim": "functional affect (not subjective experience)", "lineage_genome": wgh}
            break
    evidence = [{"cid": c.cid, "levels": sorted(c.passed)} for c in u.archive.zoo.values()
                if c.passed and hashlib.sha256(G.cpu().numpy().astype(np.float32).tobytes()).hexdigest()[:16] == c.cid[1:]]
    kids = int((pool.alive & (pool.parent == aid)).sum())
    tech = int(pool.tech[s])
    return to_jsonable({
        "aid": aid, "alive": True, "genome_id": gh, "body_id": f"B{gh[:8]}-{aid}", "generation": int(pool.generation[s]),
        "birth_tick": int(pool.birth[s]), "age": float(pool.age[s]), "parent": int(pool.parent[s]), "root": int(pool.root[s]),
        "ancestors": u.lineage.ancestors(aid, 12), "living_children": kids,
        "children_total": u.lineage.children_count.get(aid, 0), "pos": pool.pos[s].cpu().numpy().tolist(),
        "fidelity": int(pool.fidelity[s]), "energy": float(pool.energy[s]), "integrity": float(pool.integrity[s]),
        "fatigue": float(pool.fatigue[s]), "carried": float(pool.carried[s]),
        "body": {"speed": float(cap[0]), "cost": float(cap[1]), "nodes": int(cap[2]), "turn": float(cap[3]),
                 "gait_angle": float(cap[4]), "wobble": float(cap[5]), "measured": bool(pool.body_measured[s])},
        "learning": {"weight_change_relative": learned, "note": "|W − W₀| / |W₀| (aprendizado na vida, não genético)"},
        "internal_state": pool.x[s].cpu().numpy().tolist(), "modulation": float(pool.mod[s]),
        "actions": {ACTION_NAMES[i]: float(v) for i, v in enumerate(torch.sigmoid(pool.out[s]).cpu().numpy().tolist())},
        "lifetime": {"eaten": float(pool.stats[s, 0]), "distance": float(pool.stats[s, 1]), "placed": float(pool.stats[s, 2]),
                     "picked": float(pool.stats[s, 3]), "given": float(pool.stats[s, 4]), "taken": float(pool.stats[s, 5]),
                     "offspring": float(pool.stats[s, 6]), "damage": float(pool.stats[s, 7])},
        "techniques": [TECH_NAMES[b] for b in range(N_TECH) if (tech >> b) & 1],
        "culture_type": int(pool.culture_type[s]), "social": ppos, "affect": affect, "evidence": evidence,
        "species": int(pool.species[s])})


def development_replay(u: "UniverseEngine", aid: int) -> Optional[Dict[str, Any]]:
    pool = u.pool
    hit = torch.nonzero(pool.alive & (pool.aid == aid)).flatten()
    if hit.numel() == 0:
        return None
    s = int(hit[0])
    bp = u.dev.develop(pool.genome[s:s + 1], pool.aid[s:s + 1] % 100003, record=True)
    frames = [f[0].cpu().numpy().astype(np.uint8) for f in bp.frames]
    return {"aid": aid, "D": u.dev.D, "frames": [np.packbits(f).tobytes() for f in frames], "N": u.dev.N,
            "parent_cell": bp.parent_cell[0].cpu().numpy().astype(np.int16).tobytes(),
            "birth_step": bp.birth_step[0].cpu().numpy().astype(np.int16).tobytes(),
            "props": (bp.props[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().tobytes(), "prop_names": PROP_NAMES}


def search_world(u: "UniverseEngine", q: str) -> Dict[str, Any]:
    q = q.strip()[:64]
    out: Dict[str, Any] = {"q": q, "results": []}
    if re.fullmatch(r"\d{1,15}", q):
        aid = int(q)
        hit = torch.nonzero(u.pool.alive & (u.pool.aid == aid)).flatten()
        if hit.numel():
            out["results"].append({"type": "agent", "id": aid, "pos": u.pool.pos[int(hit[0])].cpu().numpy().tolist()})
        sid = int(q)
        if sid in u.settlements.settlements:
            out["results"].append({"type": "settlement", "id": sid, "pos": u.settlements.settlements[sid].get("centre")})
    if q.startswith("C") and q in u.archive.zoo:
        out["results"].append({"type": "candidate", "id": q, "levels": sorted(u.archive.zoo[q].passed)})
    if q.startswith("D"):
        for d in u.archive.discoveries:
            if d["id"].startswith(q):
                out["results"].append({"type": "discovery", "id": d["id"], "level": d["level"]})
    if q.startswith("G") or re.fullmatch(r"[0-9a-f]{6,24}", q):
        pref = q[1:] if q.startswith("G") else q
        idx = torch.nonzero(u.pool.alive).flatten()[:20000]
        for s in idx.tolist():
            if genome_hash(u.pool.genome[s:s + 1]).startswith(pref):
                out["results"].append({"type": "genome", "id": genome_hash(u.pool.genome[s:s + 1]),
                                       "agent": int(u.pool.aid[s]), "pos": u.pool.pos[s].cpu().numpy().tolist()})
                break
    return out


def ladder_status(u: "UniverseEngine") -> Dict[str, Any]:
    zoo = list(u.archive.zoo.values())

    def best(code):
        hits = [(c.cid, c.passed[code]) for c in zoo if code in c.passed]
        if not hits:
            return {"validated": False}
        cid, v = max(hits, key=lambda x: abs(x[1].get("effect", 0) or 0))
        return {"validated": True, "candidate": cid, "effect": v.get("effect"), "discovery": v.get("discovery"),
                "claim": v.get("claim")}
    tested = collections.Counter(code for c in zoo for code in c.attempts)
    return {"H": [{"code": l.split()[0], "name": l, "tested": tested.get(l.split()[0], 0), **best(l.split()[0])} for l in H_LEVELS],
            "A": [{"code": l.split()[0], "name": l, "tested": tested.get(l.split()[0], 0), **best(l.split()[0])} for l in A_LEVELS],
            "C": [{"code": l.split()[0], "name": l, "achieved": l in u.civ.achieved,
                   "tick": u.civ.achieved.get(l, {}).get("tick"),
                   "now": bool(u.civ.last_metrics.get("criteria_now", {}).get(l, False))} for l in C_LEVELS],
            "best": u.scientist.best_level, "target": u.scientist.target_level(), "c_level": u.civ.level,
            "note": "níveis = evidência por probes com controles; não são afirmações de vida real, consciência ou humanidade"}
