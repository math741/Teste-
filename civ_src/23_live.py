# %% [markdown]
# ## 23 — LIVE RESEARCH: laço único (mundo ↔ ciência ↔ evolução ↔ telemetria ↔ checkpoint)
# Orçamento adaptativo por fatias de tempo: fila científica cheia → mais validação; vazia → mais exploração (QD).
# Em `SCIENTIFIC_REPLAY` a agenda é por ticks (determinística), não por relógio. Qualquer exceção tenta checkpoint
# seguro + crash report (stack trace, versões, hardware, últimas ações); OOM → registra, limpa cache, reduz lotes,
# reduz LOD do viewer e continua. Painel compacto no notebook (o observatório 3D é a interface principal).

# %%
class LiveResearch:
    def __init__(self, uni: UniverseEngine, ck: CheckpointEngine, tel: TelemetryEngine, desk: ObserverDesk,
                 cfg: CivConfig, paths: CivPaths, logger: EventLogger, links: Dict[str, Any]):
        self.u, self.ck, self.tel, self.desk, self.cfg, self.paths, self.logger, self.links = uni, ck, tel, desk, cfg, paths, logger, links
        self.qd_gen: Optional[Generator] = None
        self.fork_gen: Optional[Generator] = None
        self.dsl_gen: Optional[Generator] = None
        self.last_ckpt = time.time()
        self.last_panel = 0.0
        self.t_start = time.time()
        self.ticks_start = uni.tick_count
        self.actions: "collections.deque" = collections.deque(maxlen=50)
        self.crashes = 0
        self.handle = None
        tel.lab_view_fn = uni.runner.view

    def _admin(self):
        while True:
            try:
                a = self.desk.admin_queue.get_nowait()
            except queue.Empty:
                return
            act, data = a["action"], a["data"]
            self.actions.append((iso_now(), "admin", act))
            rec = {"tick": self.u.tick_count, "action": act, "data": data, "ts": iso_now(), "source": "admin"}
            if act == "pause":
                self.u.paused = True
            elif act == "resume":
                self.u.paused = False
            elif act == "checkpoint":
                self.ck.save(self.u, reason="admin")
            elif act == "focus":
                self.u.user_focus_aid = int(data["aid"])
                self.u.interventions.append(dict(rec, note="promoção F4 pedida pelo usuário (registrada)"))
            elif act == "unfocus":
                self.u.user_focus_aid = None
            elif act == "fork" and self.fork_gen is None:
                self.fork_gen = self.u.counterfactual_gen(data["intervention"], data["ticks"], label="admin")
            elif act == "dsl" and self.dsl_gen is None:
                self.dsl_gen = self.u.scientist.dsl_gen(data["program"], self.u.runner)
                self.u.interventions.append(dict(rec, note="experimento DSL em arenas (laboratório)"))
            self.u.history.add("external_intervention" if act in ("focus", "fork", "dsl") else "admin_action", self.u.tick_count, action=act)

    def _advance(self, g: Optional[Generator], budget_s: float):
        if g is None:
            return None, None
        t0 = time.time()
        try:
            while time.time() - t0 < budget_s:
                next(g)
        except StopIteration as e:
            return None, e.value
        return g, None

    def _science(self, budget_s: float):
        sc = self.u.scientist
        frac_val = min(0.85, 0.35 + len(sc.queue) / max(1, self.cfg.max_queue))
        if self.fork_gen is not None:
            self.fork_gen, r = self._advance(self.fork_gen, budget_s * 0.3)
            if r is not None:
                self.actions.append((iso_now(), "fork_done", r["id"]))
        if self.dsl_gen is not None:
            self.dsl_gen, r = self._advance(self.dsl_gen, budget_s * 0.3)
        self.tel.state_label = "EXPERIMENTING" if sc.current or sc.queue else "SEARCHING"
        rep = sc.step(budget_s * frac_val)
        if rep:
            self.actions.append((iso_now(), "probe", f"{rep['code']} {rep['kind']} {'✔' if rep['passed'] else '✘'}"))
        if self.qd_gen is None:
            self.qd_gen = self.u.qd.generation_gen()
        self.qd_gen, r = self._advance(self.qd_gen, budget_s * (1 - frac_val))
        if r is not None:
            self.actions.append((iso_now(), "qd", f"ilha {r.get('island')} ger {r.get('gen', '')}"))

    def panel(self, final: bool = False):
        st = self.desk.status or {}
        u = self.u
        el = time.time() - self.t_start
        sc = u.scientist.status()
        civ = u.last.get("civ", {})
        rows = [("estado", st.get("state", "—") + (" · PAUSED" if u.paused else "")), ("tick (tempo artificial)", f"{u.tick_count:,}"),
                ("velocidade", f"{st.get('tps', 0)} ticks/s"), ("população", f"{u.pool.n_alive():,} + F0 {u.f0.total() if u.f0 else 0:.0f}"),
                ("corpos F2/F3/F4", " / ".join(str(v) for v in st.get("bodies", {}).values())),
                ("níveis H · A · C", f"{st.get('h_name', '—')} · {st.get('a_name', '—')} · {st.get('c_name', '—')}"),
                ("Scientist", f"{sc['phase']} · fila {sc['queue']} · alvo {sc['target_level']} · feitos {sc['completed']}"),
                ("descobertas", str(len(u.archive.discoveries))), ("QD", f"{u.qd.generations} gerações · cobertura {u.qd.map.coverage():.2f}"),
                ("assentamentos", f"{civ.get('settlements', 0)} · blocos {civ.get('blocks', 0):.0f}"),
                ("checkpoint", f"{self.ck.last_info.get('ts', '—')} ({self.ck.saves} nesta sessão)"),
                ("sessão / total", f"{fmt_hms(el)} / {fmt_hms(u.total_runtime + el)}")]
        acts = "".join(f"<div>{_html.escape(t[11:19])} · {_html.escape(k)} · {_html.escape(str(v))}</div>" for t, k, v in list(self.actions)[-5:])
        link = self.links.get("remote") or self.links.get("local") or ""
        html_ = ("<div style='font-family:system-ui;background:#070b14;color:#cfe0ff;padding:10px 14px;border-radius:12px;max-width:920px'>"
                 f"<div style='font-weight:700;letter-spacing:.12em'>GENESIS CIVILIZATION Ω∞ {'— SESSÃO ENCERRADA' if final else '— AO VIVO'}"
                 f"<span style='float:right;font-weight:400;font-size:12px'><a style='color:#7cf0c8' target='_blank' href='{_html.escape(link)}'>abrir mundo 3D</a></span></div>"
                 "<table style='font-size:12.5px;margin-top:6px'>" + "".join(f"<tr><td style='color:#7f8ba3;padding-right:14px'>{k}</td><td>{_html.escape(str(v))}</td></tr>" for k, v in rows)
                 + f"</table><div style='font-size:11px;color:#7f8ba3;margin-top:6px'>{acts}</div></div>")
        if self.handle is None:
            self.handle = display(HTML(html_), display_id=True)
        else:
            try:
                self.handle.update(HTML(html_))
            except Exception:
                pass

    def _crash(self, e: BaseException):
        self.crashes += 1
        rep = {"ts": iso_now(), "error": repr(e), "trace": traceback.format_exc(), "versions": ENGINE_VERSIONS,
               "hardware": self.u.hw, "tick": self.u.tick_count, "last_actions": list(self.actions)[-20:],
               "profiler": self.u.prof.report()}
        p = os.path.join(self.paths.crash_reports, f"crash_{int(time.time())}.json")
        with open(p, "w") as f:
            json.dump(to_jsonable(rep), f, indent=1)
        self.logger.event("crash", echo=True, error=repr(e), report=p)

    def run(self, max_hours: float, max_ticks: int = 0):
        cfg, u = self.cfg, self.u
        deadline = time.time() + max_hours * 3600
        self.tel.state_label = "SEARCHING"
        stop_reason = "time_limit"
        try:
            while time.time() < deadline:
                if os.path.exists(self.paths.stop_file):
                    stop_reason = "stop_file"; os.remove(self.paths.stop_file); break
                if max_ticks and u.tick_count - self.ticks_start >= max_ticks:
                    stop_reason = "max_ticks"; break
                try:
                    self._admin()
                    frac = u.scientist.adapt_budget()
                    if cfg.scientific_replay:
                        if not u.paused:
                            for _ in range(20):
                                u.tick()
                        self._science(0.02)                     # agenda por ticks: proporção fixa
                    else:
                        slice_s = 0.25
                        t0 = time.time()
                        if not u.paused:
                            while time.time() - t0 < slice_s * (1 - frac):
                                u.tick()
                        else:
                            time.sleep(0.05)
                        self._science(slice_s * frac)
                    self.tel.pump()
                    if time.time() - self.last_ckpt >= cfg.checkpoint_every_minutes * 60:
                        self.tel.state_label = "CHECKPOINTING"
                        self.ck.save(u, reason="periodic"); self.last_ckpt = time.time()
                        self.tel.checkpoint_info = self.ck.last_info
                    if self.ck.last_info:
                        self.tel.checkpoint_info = self.ck.last_info
                    if time.time() - self.last_panel >= 1.0:
                        self.panel(); self.last_panel = time.time()
                    if cfg.stop_on_champion and u.archive.overall_champion:
                        stop_reason = "champion"; break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if is_oom_error(e):
                        self.logger.event("oom", echo=True, error=repr(e)[:300])
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        oom_backoff(cfg)
                        u.scientist._gen = None; u.scientist.current = None
                        self.qd_gen = None
                        continue
                    self._crash(e)
                    try:
                        self.ck.save(u, sync=True, reason="after_exception")
                    except Exception as e2:
                        self.logger.event("checkpoint_after_crash_failed", error=repr(e2))
                    u.scientist._gen = None; u.scientist.current = None; self.qd_gen = None
                    if self.crashes > 20:
                        stop_reason = "too_many_errors"; break
        except KeyboardInterrupt:
            stop_reason = "user_interrupt"
        finally:
            self.tel.state_label = "CHECKPOINTING"
            self.tel.pump(force=True)
            u.total_runtime += time.time() - self.t_start
            self.ck.wait()
            self.ck.save(u, sync=True, reason=f"session_end:{stop_reason}")
            self.tel.checkpoint_info = self.ck.last_info
            self.tel.state_label = "OFFLINE"
            self.desk.online = False
            self.tel.pump(force=True)
            u.history.flush(); self.logger.flush()
            self.panel(final=True)
        return stop_reason
