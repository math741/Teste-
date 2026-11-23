# %% [markdown]
# ## 12b. Painel AO VIVO — atualiza a cada segundo, sem parar a pesquisa
# Enquanto a pesquisa roda, a saída da célula de pesquisa vira um painel que se atualiza sozinho:
# * **agora no simulador**: mosaico de universos do lote que está sendo simulado neste instante (PCA dos canais → cor);
# * **fase atual** com barra de progresso (triagem, arquivos, experimento do Scientist, ChampionLab, checkpoint…);
# * contadores, velocidade, **melhor candidato** com scorecard e nível na escada, anúncios de descoberta, eventos recentes;
# * **gráficos por geração**: convergência, MAP-Elites, escada de descoberta, zoológico, operadores de mutação.
#
# Não é preciso interromper nada para ver resultados. O painel é só observação: lê o estado, nunca participa do
# fitness, da seleção ou dos testes, e não altera nenhum gerador aleatório da pesquisa.
# Fora de notebook (script), o painel vira uma linha de status periódica.

# %%
LIVE_STATUS_COLORS = {STATUS_PASS: "#3fb950", STATUS_FAIL: "#f85149", STATUS_INCONCLUSIVE: "#d29922",
                      STATUS_ERROR: "#db6d28", STATUS_NOT_TESTED: "#6e7681", STATUS_RUNNING: "#58a6ff"}
LEVEL_COLORS = ["#6e7681", "#8b949e", "#58a6ff", "#a371f7", "#db61a2", "#f0883e", "#d29922", "#3fb950", "#56d364",
                "#ffd33d", "#ffffff"]
PT_FIELD = {"activity": "atividade", "persistence": "persistência", "mobility": "mobilidade", "complexity": "complexidade",
            "causality": "causalidade", "memory": "memória", "channel": "canal", "transformation": "transformação",
            "composition": "composição", "generalization": "generalização", "recovery": "recuperação",
            "robustness": "robustez", "adaptation": "adaptação", "replication": "replicação", "novelty": "novidade"}
_ROOT = ("background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:12px;padding:12px 14px;margin:4px 0;"
         "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.45;")
_MUTED = "color:#8b949e;"


def _esc(x) -> str:
    return _html.escape(str(x))


def _short(gid: Optional[str]) -> str:
    return "-" if not gid else str(gid)


def _human(n: float) -> str:
    n = float(n)
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def rgb_tile(U: np.ndarray) -> np.ndarray:
    """[C,S,S] -> RGB uint8 por PCA dos canais (sinal fixado para as cores não piscarem entre quadros)."""
    C, S, _ = U.shape
    X = np.nan_to_num(U.reshape(C, -1).T.astype(np.float64))
    X = X - X.mean(0, keepdims=True)
    if np.abs(X).max() < 1e-9:
        return np.zeros((S, S, 3), np.uint8)
    k = min(3, C)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:k]
        V = V * np.sign(V[np.arange(k), np.abs(V).argmax(1)])[:, None]
        Y = X @ V.T
    except np.linalg.LinAlgError:
        Y = X[:, :k]
    if Y.shape[1] < 3:
        Y = np.hstack([Y, np.zeros((Y.shape[0], 3 - Y.shape[1]))])
    lo, hi = np.percentile(Y, 1), np.percentile(Y, 99)
    Y = np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1)
    return (Y.reshape(S, S, 3) * 255).astype(np.uint8)


def mosaic_jpeg(frames: List[np.ndarray], cols: int = 4, tile: int = 112, quality: int = 78) -> bytes:
    from PIL import Image
    n = max(1, len(frames))
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))
    gap = 3
    canvas = Image.new("RGB", (cols * tile + (cols - 1) * gap, rows * tile + (rows - 1) * gap), (13, 17, 23))
    for i, f in enumerate(frames):
        im = Image.fromarray(rgb_tile(f)).resize((tile, tile), Image.NEAREST)
        canvas.paste(im, ((i % cols) * (tile + gap), (i // cols) * (tile + gap)))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def svg_spark(vals: Sequence[float], label: str, color: str = "#58a6ff", fmt: str = "{:.3f}", w: int = 200,
              h: int = 44) -> str:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)][-300:]
    head = f'<div style="font-size:11px;{_MUTED}">{_esc(label)} <b style="color:#e6edf3">{fmt.format(v[-1]) if v else "–"}</b></div>'
    if len(v) < 2:
        return (f'<div style="margin:4px 10px 4px 0">{head}<div style="height:{h}px;width:{w}px;background:#161b22;'
                f'border-radius:4px;color:#6e7681;font-size:10.5px;padding:4px">aguardando dados…</div></div>')
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    n = len(v)
    pts = " ".join(f"{1 + i * (w - 2) / (n - 1):.1f},{h - 3 - (x - lo) / (hi - lo) * (h - 6):.1f}" for i, x in enumerate(v))
    return (f'<div style="margin:4px 10px 4px 0">{head}<svg width="{w}" height="{h}" style="background:#161b22;border-radius:4px">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/></svg></div>')


def svg_map(me: MapElites, cell: int = 15) -> str:
    import matplotlib.colors as mcolors
    H = me.heat()
    nx, ny = H.shape
    vals = H[np.isfinite(H)]
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    rects = []
    for i in range(nx):
        for j in range(ny):
            v = H[i, j]
            col = "#161b22" if not np.isfinite(v) else mcolors.to_hex(plt.cm.viridis((v - lo) / (hi - lo + 1e-9)))
            rects.append(f'<rect x="{i * cell}" y="{(ny - 1 - j) * cell}" width="{cell - 1}" height="{cell - 1}" fill="{col}"/>')
    return (f'<div><div style="font-size:11px;{_MUTED}">MAP-Elites — {_esc(me.axes[0])} → / {_esc(me.axes[1])} ↑ · '
            f'cobertura <b style="color:#e6edf3">{me.coverage():.1%}</b></div>'
            f'<svg width="{nx * cell}" height="{ny * cell}" style="background:#0d1117">{"".join(rects)}</svg></div>')


def _chip(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;margin:2px 3px 2px 0;padding:1px 7px;border-radius:10px;font-size:11px;'
            f'border:1px solid {color};color:{color}">{text}</span>')


def _level_badge(level: int) -> str:
    lv = max(-1, min(10, int(level)))
    name = LEVEL_NAMES.get(lv, "inviável") if lv >= 0 else "inviável"
    col = LEVEL_COLORS[lv] if lv >= 0 else "#f85149"
    return (f'<span style="padding:1px 8px;border-radius:10px;background:{col};color:#0d1117;font-weight:700;'
            f'font-size:11px">L{lv} {_esc(name)}</span>')


class LiveMonitor:
    """Painel que se redesenha a cada `interval` segundos, alimentado pelos próprios passos da simulação."""

    def __init__(self, lab: "GenesisLab", interval: float = 1.0, refresh_every: float = 90.0, text_every: float = 15.0):
        self.lab = lab
        self.interval = max(0.25, float(interval))
        self.refresh_every = refresh_every
        self.text_every = text_every
        self.mode = "html" if IN_NOTEBOOK else "text"
        self.status = "SEARCHING"
        self.phase, self.detail, self.done, self.total = "iniciando", "", 0.0, 0.0
        self.phase_t0 = time.time()
        self.phase_steps0 = lab.sim.steps_done
        self.last = 0.0
        self.last_text = 0.0
        self.last_refresh = 0.0
        self.frames: Optional[np.ndarray] = None
        self.frames_info = ""
        self.frames_jpeg_b64: Optional[str] = None
        self.frames_dirty = False
        self.best_cache: Tuple[Optional[str], Optional[str], float] = (None, None, 0.0)
        self.view_ts = 0.0
        self.charts_key: Any = None
        self.charts_ts = 0.0
        self.speed = 0.0
        self.speed_hist: "collections.deque" = collections.deque(maxlen=240)
        self._steps_prev, self._steps_t = lab.sim.steps_done, time.time()
        self.banners: List[Tuple[float, str]] = []
        self.handles: Dict[str, Any] = {}
        self.viz_sim: Optional[Simulator] = None
        self.errors = 0
        self._busy = False

    # ------------------------------------------------------------------ entradas
    def start(self):
        self.render(force=True)

    def on_step(self, sim: "Simulator", gb: "GenomeBatch", st: "SimState"):
        if self._busy or sim is self.viz_sim:
            return
        now = time.time()
        if now - self.last < self.interval:
            return
        self._capture(st)
        self.render()

    def progress(self, phase=None, done=None, total=None, detail=None):
        if phase is not None and phase != self.phase:
            self.phase, self.phase_t0 = phase, time.time()
            self.phase_steps0 = self.lab.sim.steps_done
            self.done, self.total, self.detail = 0.0, 0.0, ""
        if detail is not None:
            self.detail = detail
        if total is not None:
            self.total = float(total)
        if done is not None:
            self.done = float(done)
        if not self._busy and time.time() - self.last >= self.interval:
            self.render()

    def announce(self, text: str):
        self.banners.append((time.time(), text))
        self.banners = self.banners[-4:]

    def finish(self, status: str):
        self.status = status
        self.phase, self.detail, self.done, self.total = status, "", 0.0, 0.0
        if self.mode == "html":
            self._busy = True
            try:
                self._open_handles(time.time())      # painel final sempre visível (saídas novas)
            except Exception as e:
                print(f"[painel] não foi possível desenhar o estado final: {e!r}")
            finally:
                self._busy = False
        else:
            self.render(force=True)

    # ------------------------------------------------------------------ captura
    def _capture(self, st: "SimState"):
        try:
            B = int(st.U.shape[0])
            pool = B // 2 if B > 16 else B
            k = min(8, pool)
            idx = np.unique(np.linspace(0, pool - 1, k).astype(int))
            self.frames = st.U[torch.as_tensor(idx, device=st.U.device)].float().cpu().numpy()
            self.frames_info = f"{len(idx)} de {B} universos do lote atual · passo interno t={st.t}"
            self.frames_dirty = True
        except Exception as e:
            self.frames_info = f"captura indisponível ({type(e).__name__})"

    # ------------------------------------------------------------------ desenho
    def render(self, force: bool = False):
        if self._busy:
            return
        self._busy = True
        try:
            now = time.time()
            self.last = now
            self._update_speed(now)
            if self.mode == "text":
                if force or now - self.last_text >= self.text_every:
                    print(self._text_line(), flush=True)
                    self.last_text = now
                return
            if not self.handles or now - self.last_refresh >= self.refresh_every:
                self._open_handles(now)
                return
            self.handles["head"].update(HTML(self._head_html(now)))
            best = self.lab.best_candidate()
            if force or self.frames_dirty or best != self.best_cache[0] or now - self.view_ts > 5:
                self.handles["view"].update(HTML(self._view_html(now, best)))
                self.view_ts = now
            key = (self.lab.counters["generation"], len(self.lab.results), self.lab.counters["discovery_events"],
                   len(self.lab.map.cells))
            if force or key != self.charts_key or now - self.charts_ts > 20:
                self.handles["charts"].update(HTML(self._charts_html()))
                self.charts_key, self.charts_ts = key, now
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.errors += 1
            if self.errors <= 3:
                self.lab.logger.event("live_render_error", error=repr(e), tb=traceback.format_exc()[-1200:])
            if self.errors >= 25:
                self.mode = "text"     # degrada para linha de status, sem nunca derrubar a pesquisa
        finally:
            self._busy = False

    def _open_handles(self, now: float):
        """(Re)cria as saídas. Periodicamente refeito: se o navegador recarregar a página, o painel volta sozinho."""
        if self.handles:
            clear_output(wait=True)
        best = self.lab.best_candidate()
        self.handles = {"head": display(HTML(self._head_html(now)), display_id=True),
                        "view": display(HTML(self._view_html(now, best)), display_id=True),
                        "charts": display(HTML(self._charts_html()), display_id=True)}
        self.last_refresh = self.view_ts = self.charts_ts = now

    def _update_speed(self, now: float):
        s = self.lab.sim.steps_done
        dt = now - self._steps_t
        if dt >= 0.5:
            self.speed = (s - self._steps_prev) / dt
            self._steps_prev, self._steps_t = s, now
            self.speed_hist.append(self.speed)

    def _text_line(self) -> str:
        lab = self.lab
        c = lab.counters
        best = lab.best_candidate()
        pct = f" {100 * self.done / self.total:.0f}%" if self.total else ""
        return (f"[{fmt_hms(time.time() - lab.session_start)}] gen {c['generation']} | {c['evaluations']:,} universos | "
                f"{self.speed:,.0f} u·passos/s | {self.phase}{pct} | melhor L{lab.levels.get(best, -1) if best else -1} {best} | "
                f"exp {c['experiments']} | descobertas {c['discovery_events']}")

    def _tile(self, label: str, value: str, sub: str = "") -> str:
        return (f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:6px 9px">'
                f'<div style="{_MUTED}font-size:10px;text-transform:uppercase;letter-spacing:.05em">{_esc(label)}</div>'
                f'<div style="font-size:16px;font-weight:600">{value}</div>'
                f'<div style="{_MUTED}font-size:10.5px">{sub}</div></div>')

    def _head_html(self, now: float) -> str:
        lab = self.lab
        c = lab.counters
        best = lab.best_candidate()
        best_lv = lab.levels.get(best, -1) if best else -1
        last_gen = lab.history[-1] if lab.history else {}
        st_col = "#3fb950" if self.status == "SEARCHING" else "#d29922"
        last_ck = lab.ckpt.last_save_ts
        nxt = max(0.0, lab.cfg.checkpoint_every_minutes * 60 - (now - lab.last_checkpoint_ts))
        avg_speed = float(np.mean(self.speed_hist)) if self.speed_hist else 0.0
        tiles = "".join([
            self._tile("Geração", f"{c['generation']:,}", f"{last_gen.get('gen_seconds', 0):.0f} s na última"),
            self._tile("Universos avaliados", f"{c['evaluations']:,}", f"{lab.throughput['evals_per_hour_est']:,.0f}/h (estim.)"),
            self._tile("Passos simulados", _human(c["sim_steps"]), "universo·passos"),
            self._tile("Velocidade", f"{_human(self.speed)}/s", f"média {_human(avg_speed)}/s"),
            self._tile("Melhor nível", _level_badge(best_lv), f"recorde L{lab.best_level_global}"),
            self._tile("Descobertas", f"{c['discovery_events']:,}", f"experimentos {c['experiments']:,}"),
            self._tile("🎯 Foco (próximo degrau)", _level_badge(lab.focus_level()) if lab.cfg.directed_search else "aberta",
                       f"medida {_esc(lab.focus_key())}: {lab.focus_stats()['focus_proxy_best']:.2f}" if lab.cfg.directed_search
                       else "busca não dirigida"),
            self._tile("Fila do Scientist", f"{len(lab.scientist.queue):,}", f"testes do foco: {lab.focus_tests_done:,}"),
            self._tile("Espécies", f"{len(lab.species.reps):,}", f"{len(lab.species.extinct):,} extintas"),
            self._tile("MAP-Elites", f"{lab.map.coverage():.1%}", f"{len(lab.map.cells)} células"),
            self._tile("Anomalias", f"{len(lab.anomalies.entries):,}", f"falhas arquivadas {len(lab.failures.entries):,}"),
            self._tile("Novelty zoo", f"{len(lab.novelty.ids):,}", f"campeões provisórios {len(lab.champions.entries)}"),
            self._tile("Checkpoint", datetime.datetime.fromtimestamp(last_ck).strftime("%H:%M:%S") if last_ck else "-",
                       f"próximo em {fmt_hms(nxt)}"),
        ])
        if self.status != "SEARCHING":
            bar, prog = "", "pesquisa parada — estado salvo"
        elif self.total > 0:
            pct = max(0.0, min(100.0, 100.0 * self.done / self.total))
            bar = (f'<div style="background:#21262d;border-radius:6px;height:10px;margin:4px 0">'
                   f'<div style="width:{pct:.1f}%;background:#58a6ff;height:10px;border-radius:6px"></div></div>')
            prog = f"{pct:.0f}%"
        else:
            bar = ('<div style="background:#21262d;border-radius:6px;height:10px;margin:4px 0">'
                   '<div style="width:100%;background:#1f6feb55;height:10px;border-radius:6px"></div></div>')
            prog = "em andamento"
        phase_steps = self.lab.sim.steps_done - self.phase_steps0
        directed = ""
        if lab.cfg.directed_search:
            fl = lab.focus_level()
            pst = lab.scientist.stats.get(FOCUS_PROBE.get(fl, ""), {"n": 0, "pass": 0})
            directed = (f'<div style="margin:6px 0;padding:5px 8px;border:1px solid #1f6feb;border-radius:8px;background:#0c2d6b55">'
                        f'🎯 <b>Busca dirigida</b> — foco <b>L{fl} {_esc(LEVEL_NAMES.get(fl, ""))}</b> '
                        f'(probe <b>{_esc(FOCUS_PROBE.get(fl, "-"))}</b>: {pst["n"]} testes, {pst["pass"]} PASS) · '
                        f'medida graduada <b>{_esc(lab.focus_key())}</b>: melhor {lab.focus_stats()["focus_proxy_best"]:.2f} '
                        f'· {len(lab.frontier(16))} universos na fronteira geram ~{int(100 * max(0.2, 0.5 - 0.06 * lab.stagnation["stage"]))}% dos filhos</div>')
        events = self._events_html()
        where = f"{_esc(lab.paths.checkpoints)} ({'Google Drive' if lab.paths.on_drive else 'disco local do Colab'})"
        return (f'<div style="{_ROOT}">'
                f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;align-items:baseline">'
                f'<div style="font-size:15px;font-weight:700"><span style="color:{st_col}">●</span> PROJECT GENESIS Ω '
                f'<span style="color:{st_col}">AO VIVO</span> <span style="{_MUTED}font-weight:400;font-size:11px">'
                f'atualiza a cada {self.interval:g} s · {datetime.datetime.now().strftime("%H:%M:%S")}</span></div>'
                f'<div style="{_MUTED}">sessão {fmt_hms(now - lab.session_start)} · tempo total {fmt_hms(lab.runtime_total())} · '
                f'sessão nº {c["sessions"]} · <b style="color:{st_col}">{_esc(self.status)}</b></div></div>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(138px,1fr));gap:6px;margin:10px 0">{tiles}</div>'
                f'{directed}<div>▶ <b>{_esc(self.phase)}</b> <span style="{_MUTED}">{_esc(self.detail)}</span></div>{bar}'
                f'<div style="{_MUTED}font-size:11px">{prog} · {fmt_hms(now - self.phase_t0)} nesta fase · '
                f'{_human(phase_steps)} universo·passos nesta fase</div>'
                f'<div style="margin-top:8px">{events}</div>'
                f'<div style="{_MUTED}font-size:11px;margin-top:8px;border-top:1px solid #21262d;padding-top:6px">'
                f'💾 {where} · física {"v2 (taxa local)" if lab.cfg.local_rate_limit else "v1 (dt global)"} · '
                f'⏸ pausar: botão ■ da célula (salva tudo) · ▶ continuar: rode esta célula de novo '
                f'(ou Executar tudo) — a pesquisa retoma do último checkpoint</div></div>')

    def _events_html(self) -> str:
        rows = []
        for e in reversed(self.lab.logger.recent[-80:]):
            line = self._fmt_event(e)
            if line:
                rows.append(f'<div style="font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                            f'<span style="{_MUTED}">{_esc(e.get("ts", "")[11:19])}</span> {line}</div>')
            if len(rows) >= 9:
                break
        if not rows:
            rows = [f'<div style="{_MUTED}font-size:11.5px">aguardando os primeiros eventos…</div>']
        return f'<div style="{_MUTED}font-size:11px;margin-bottom:2px">EVENTOS RECENTES</div>' + "".join(rows)

    @staticmethod
    def _fmt_event(e: Dict[str, Any]) -> str:
        k = e.get("kind")
        if k == "experiment":
            col = LIVE_STATUS_COLORS.get(e.get("status"), "#e6edf3")
            sc = e.get("score")
            sc_s = f" {sc:.2f}" if isinstance(sc, (int, float)) and sc is not None and np.isfinite(sc) else ""
            return (f'🧪 {_esc(e.get("probe"))} → <b style="color:{col}">{_esc(e.get("status"))}{sc_s}</b> em '
                    f'{_esc(e.get("candidate"))} <span style="{_MUTED}">({e.get("seconds")} s)</span>')
        if k == "discovery":
            return (f'⭐ <b style="color:#ffd33d">descoberta L{e.get("level")} {_esc(LEVEL_NAMES.get(e.get("level"), ""))}</b> '
                    f'via {_esc(e.get("probe"))} em {_esc(e.get("candidate"))}')
        if k == "anomaly":
            return f'⚠️ anomalia {_esc(e.get("genome_id"))} (surpresa {e.get("surprise", 0):.1f})'
        if k == "checkpoint":
            return f'💾 checkpoint salvo (geração {e.get("generation")}, {_esc(e.get("reason"))})'
        if k == "stagnation":
            return f'🌀 estagnação, estágio {e.get("stage")}: {_esc(e.get("action"))}'
        if k == "curriculum":
            return f'📈 currículo: {_esc(e.get("change"))} → {e.get("value")}'
        if k == "champion_lab":
            return f'🏆 ChampionLab {_esc(e.get("candidate"))}: {"PASSOU" if e.get("passed") else "não passou"}'
        if k == "GENESIS_CHAMPION":
            return f'🏆 <b style="color:#ffd33d">GENESIS CHAMPION {_esc(e.get("candidate"))}</b>'
        if k == "METRIC_EXPLOIT_SUSPECTED":
            return f'🚩 suspeita de exploit de métrica (geração {e.get("generation")})'
        if k == "OOM":
            return f'🧯 falta de memória na GPU: batch {e.get("old_batch")} → {e.get("new_batch")} (recuperado)'
        if k in ("resume", "research_start"):
            return f'▶ pesquisa iniciada/retomada na geração {e.get("generation")}'
        if k == "crash":
            return f'💥 erro: {_esc(e.get("error"))}'
        return ""

    def _render_best(self, cid: str) -> Optional[str]:
        g = self.lab.store.get(cid)
        if g is None:
            return None
        if self.viz_sim is None:
            self.viz_sim = Simulator(self.lab.cfg, self.lab.device)
        gb = self.viz_sim.prepare([g])
        st = self.viz_sim.init_state(gb, [self.lab.envs[0]], [derive_seed(cid, "thumb")], run_seed=1)
        frames = []
        for t in range(96):
            st = self.viz_sim.step(gb, st)
            if t in (31, 63, 95):
                frames.append(st.U[0].float().cpu().numpy())
        return base64.b64encode(mosaic_jpeg(frames, cols=3, tile=96)).decode("ascii")

    def _view_html(self, now: float, best: Optional[str]) -> str:
        lab = self.lab
        if self.frames is not None and (self.frames_dirty or self.frames_jpeg_b64 is None):
            self.frames_jpeg_b64 = base64.b64encode(mosaic_jpeg(list(self.frames))).decode("ascii")
            self.frames_dirty = False
        live = (f'<img src="data:image/jpeg;base64,{self.frames_jpeg_b64}" style="width:100%;max-width:470px;'
                f'image-rendering:pixelated;border-radius:6px"/>' if self.frames_jpeg_b64 else
                f'<div style="{_MUTED}height:120px">aguardando o primeiro lote…</div>')
        if best and best != self.best_cache[0] and (self.best_cache[0] is None or now - self.best_cache[2] > 20):
            try:
                self.best_cache = (best, self._render_best(best), now)
            except Exception as e:
                self.best_cache = (best, None, now)
                lab.logger.event("live_thumb_error", error=repr(e))
        best_html = f'<div style="{_MUTED}">ainda sem candidatos avaliados</div>'
        if best:
            sc = lab.scorecard(best)
            g = lab.store.get(best)
            chips = []
            for f in ["persistence", "complexity", "causality", "memory", "channel", "transformation", "composition",
                      "generalization", "recovery", "robustness", "adaptation", "replication"]:
                v = sc.get(f, STATUS_NOT_TESTED)
                if isinstance(v, (int, float)):
                    chips.append(_chip(f"{PT_FIELD.get(f, f)} {v:.2f}", "#58a6ff"))
                else:
                    stt, _, num = str(v).partition(":")
                    mark = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?", "ERROR": "!", "NOT_TESTED": "·"}.get(stt, "")
                    chips.append(_chip(f"{PT_FIELD.get(f, f)} {mark}{(' ' + num[:4]) if num else ''}",
                                       LIVE_STATUS_COLORS.get(stt, "#6e7681")))
            thumb = (f'<img src="data:image/jpeg;base64,{self.best_cache[1]}" style="width:100%;max-width:300px;'
                     f'image-rendering:pixelated;border-radius:6px;margin-top:6px"/>'
                     f'<div style="{_MUTED}font-size:10.5px">o melhor universo em t = 32, 64, 96</div>'
                     if self.best_cache[0] == best and self.best_cache[1] else "")
            best_html = (f'<div style="font-size:13px"><b>{_esc(best)}</b> {_level_badge(sc.get("level", -1))}</div>'
                         f'<div style="{_MUTED}font-size:11px">fitness {sc.get("quality(fitness)", 0):.3f} · evidência '
                         f'{sc.get("evidence", 0):.2f} · espécie {g.species_id if g else "-"} · geração {g.generation if g else "-"} · '
                         f'{_esc(g.operator if g else "")}</div>{thumb}<div style="margin-top:6px">{"".join(chips)}</div>')
        banners = ""
        for ts, text in reversed(self.banners):
            if now - ts < 300:
                lines = [l.strip().strip("║").strip() for l in text.splitlines()]
                body = "\n".join(l for l in lines if l and not set(l) <= set("╔╗╚╝═║ "))
                banners += (f'<div style="margin-top:8px;border:1px solid #d29922;border-radius:8px;padding:6px 8px;'
                            f'background:#2d2205;white-space:pre-wrap;font-size:11px">{_esc(body[:900])}</div>')
        return (f'<div style="{_ROOT}display:flex;flex-wrap:wrap;gap:14px">'
                f'<div style="flex:1 1 440px;min-width:280px"><div style="{_MUTED}font-size:11px;margin-bottom:4px">'
                f'🔬 AGORA NO SIMULADOR — {_esc(self.frames_info)}</div>{live}'
                f'<div style="{_MUTED}font-size:10.5px">cada quadrado é um universo artificial do lote atual '
                f'(PCA dos canais → cor). A imagem muda a cada segundo.</div></div>'
                f'<div style="flex:1 1 300px;min-width:260px"><div style="{_MUTED}font-size:11px;margin-bottom:4px">'
                f'🏅 MELHOR CANDIDATO ATÉ AGORA</div>{best_html}{banners}</div></div>')

    def _charts_html(self) -> str:
        lab = self.lab
        h = lab.history
        col = lambda k: [r.get(k) for r in h]
        sparks = "".join([
            svg_spark(col("best_quality"), "melhor fitness", "#58a6ff"),
            svg_spark(col("map_coverage"), "cobertura MAP-Elites", "#3fb950", "{:.1%}"),
            svg_spark(col("best_level"), "melhor nível", "#ffd33d", "{:.0f}"),
            svg_spark(col("focus_proxy_best"), "medida do foco (melhor)", "#1f6feb"),
            svg_spark(col("species"), "espécies", "#a371f7", "{:.0f}"),
            svg_spark(col("viable_fraction"), "fração viável", "#f0883e", "{:.1%}"),
            svg_spark(list(self.speed_hist), "velocidade (u·passos/s)", "#8b949e", "{:,.0f}"),
        ])
        counts = {L: 0 for L in range(10)}
        for l in lab.levels.values():
            if 0 <= l <= 9:
                counts[l] += 1
        mx = max(1, max(counts.values()))
        ladder = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;font-size:11px">'
            f'<span style="width:190px;{_MUTED}">L{L} {_esc(LEVEL_NAMES[L][:24])}</span>'
            f'<span style="display:inline-block;height:9px;width:{max(1, int(150 * counts[L] / mx)) if counts[L] else 0}px;'
            f'background:{LEVEL_COLORS[L]};border-radius:3px"></span><span>{counts[L]}</span></div>'
            for L in range(9, -1, -1))
        zoo = "".join(_chip(f"{k.lower()} {v}", "#8b949e") for k, v in lab.zoo.counts().items() if v)
        via = getattr(lab, "last_viability", {}) or {}
        via_cols = {"VIABLE": "#3fb950", "DEAD": "#6e7681", "FROZEN": "#58a6ff", "EXPLOSIVE": "#f85149",
                    "PURE_NOISE": "#d29922", "NUMERICALLY_UNSTABLE": "#db6d28"}
        via_html = "".join(_chip(f"{k.lower()} {v}", via_cols.get(k, "#8b949e")) for k, v in sorted(via.items(), key=lambda x: -x[1]))
        ops = "".join(_chip(f"{k} {v:.2f}", "#58a6ff") for k, v in lab.bandit.probs().items())
        return (f'<div style="{_ROOT}">'
                f'<div style="{_MUTED}font-size:11px">📈 RESULTADOS POR GERAÇÃO (atualiza a cada geração)</div>'
                f'<div style="display:flex;flex-wrap:wrap">{sparks}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:22px;margin-top:8px">'
                f'<div>{svg_map(lab.map)}</div>'
                f'<div><div style="{_MUTED}font-size:11px;margin-bottom:2px">ESCADA DE DESCOBERTA (candidatos retidos)</div>{ladder}</div>'
                f'<div style="max-width:380px"><div style="{_MUTED}font-size:11px">ÚLTIMA GERAÇÃO (sobrevivência dinâmica)</div>'
                f'{via_html or "<span style=color:#6e7681>-</span>"}'
                f'<div style="{_MUTED}font-size:11px;margin-top:6px">ZOOLÓGICO</div>{zoo or "<span style=color:#6e7681>-</span>"}'
                f'<div style="{_MUTED}font-size:11px;margin-top:6px">OPERADORES DE MUTAÇÃO (recompensa média, bandit UCB)</div>{ops}'
                f'</div></div></div>')
