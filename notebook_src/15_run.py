# %% [markdown]
# ## 15. BOOT → HARDWARE DETECTION → AUTO-TUNING
# Detecta CPU/GPU e VRAM, escolhe perfil, mede a memória real por batch (busca dobrando até o alvo de VRAM do perfil
# — EXTREME ≈ 87%), testa precisão mista e `torch.compile` (só adota se for estável **e** mais rápido) e roda o
# microbenchmark. Throughput exibido é **estimativa, não promessa**.

# %%
def autotune(cfg: GenesisConfig, hw: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"profile": cfg.profile}
    space = ParamSpace(cfg)
    env = make_environment_suite(cfg.master_seed)[0]
    rng = np.random.default_rng(0)
    proto = [Genome(space.random_params(rng)) for _ in range(32)]
    for g in proto:
        g.params["link_active"][:] = True     # pior caso de memória
        g.params["delay_w"][:] = 0.5
    sim = Simulator(cfg, device)

    def trial(B, steps=3):
        gl = (proto * (B // len(proto) + 1))[:B]
        gb = sim.prepare(gl)
        st = sim.init_state(gb, [env] * B, list(range(B)))
        st = sim.run(gb, st, steps)
        _sync()
        del gb, st

    if device.type == "cuda":
        target = VRAM_TARGET.get(cfg.profile, 0.6)
        free = torch.cuda.mem_get_info()[0]
        best, B = 16, 16
        while B <= 16384:
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                trial(B)
                peak = torch.cuda.max_memory_allocated() - base
                rep[f"peak@{B}"] = peak
                # margem para gravação de quadros da triagem e para os probes
                if peak * 1.35 > target * free:
                    break
                best = B
                B *= 2
            except Exception as e:
                if is_oom_error(e):
                    torch.cuda.empty_cache()
                    break
                raise
        torch.cuda.empty_cache()
        cfg.batch_size = int(best)
        cap = {"DEBUG": 24, "FAST": 256, "RESEARCH": 1024, "EXTREME": 2048}[cfg.profile]
        cfg.population_size = int(max(32, min(cap, best // max(1, cfg.screen_envs))))
        rep["vram_free"] = free; rep["vram_target_fraction"] = target
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        cfg.batch_size = min(cfg.batch_size, 64)
    # precisão mista: só se estável e mais rápida
    rep["precision"] = "fp32"
    if device.type == "cuda" and cfg.profile in ("RESEARCH", "EXTREME"):
        try:
            Bt = min(cfg.batch_size, 128)
            gl = (proto * (Bt // len(proto) + 1))[:Bt]
            outs, times = {}, {}
            for prec in ["fp32", "bf16" if hw.get("bf16_supported") else "fp16"]:
                c2 = copy.deepcopy(cfg); c2.precision = prec
                s2 = Simulator(c2, device); gb = s2.prepare(gl)
                st = s2.init_state(gb, [env] * Bt, list(range(Bt)), run_seed=1)
                s2.run(gb, st, 3); _sync(); t0 = time.time()
                st = s2.run(gb, st, 40); _sync()
                times[prec] = time.time() - t0; outs[prec] = st.U.float()
            lp = [k for k in outs if k != "fp32"][0]
            rel = float((outs[lp] - outs["fp32"]).abs().mean() / (outs["fp32"].abs().mean() + 1e-9))
            speed = times["fp32"] / max(1e-9, times[lp])
            rep["precision_test"] = {"candidate": lp, "rel_error": rel, "speedup": speed}
            if rel < 0.02 and speed > 1.15:
                cfg.precision = lp; rep["precision"] = lp
        except Exception as e:
            rep["precision_test"] = {"error": repr(e)}
    # torch.compile: testado automaticamente; usado só se estável e mais rápido
    rep["compile"] = False
    if device.type == "cuda" and cfg.profile in ("RESEARCH", "EXTREME") and hasattr(torch, "compile"):
        try:
            Bt = min(cfg.batch_size, 256)
            gl = (proto * (Bt // len(proto) + 1))[:Bt]
            res = {}
            for comp in [False, True]:
                c2 = copy.deepcopy(cfg); c2.use_compile = comp
                s2 = Simulator(c2, device); gb = s2.prepare(gl)
                st = s2.init_state(gb, [env] * Bt, list(range(Bt)), run_seed=1)
                s2.run(gb, st, 3); _sync(); t0 = time.time()
                st = s2.run(gb, st, 30); _sync()
                res[comp] = (time.time() - t0, st.U.clone())
            err = float((res[True][1] - res[False][1]).abs().max())
            speed = res[False][0] / max(1e-9, res[True][0])
            rep["compile_test"] = {"speedup": speed, "max_abs_diff": err}
            if speed > 1.15 and err < 1e-3:
                cfg.use_compile = True; rep["compile"] = True
        except Exception as e:
            rep["compile_test"] = {"error": repr(e)[:300]}
    # microbenchmark
    sim = Simulator(cfg, device)
    B = cfg.batch_size
    gl = (proto * (B // len(proto) + 1))[:B]
    gb = sim.prepare(gl)
    st = sim.init_state(gb, [env] * B, list(range(B)))
    sim.run(gb, st, 2); _sync()
    t0 = time.time(); n = 10
    sim.run(gb, st, n); _sync()
    dt = time.time() - t0
    ups = B * n / dt
    per_eval = 2 * cfg.screen_steps * cfg.screen_envs
    rep.update({"batch_size": cfg.batch_size, "population_size": cfg.population_size, "world_size": cfg.world_size,
                "steps_per_sec": n / dt, "universe_steps_per_sec": ups, "universes_per_sec": ups / cfg.screen_steps,
                "estimated_evaluations_per_hour": ups / per_eval * 3600,
                "snapshot_frequency_minutes": cfg.checkpoint_every_minutes})
    del gb, st
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rep


# ----------------------------------------------------------------------------
print("╔══════════════════════════════════════╗\n║          PROJECT GENESIS Ω           ║\n║               BOOT                   ║\n╚══════════════════════════════════════╝")
random.seed(MASTER_SEED); np.random.seed(MASTER_SEED % (2 ** 32)); torch.manual_seed(MASTER_SEED)
torch.set_flush_denormal(True)   # evita lentidão de subnormais em CPU (campos que decaem a ~0)
PATHS = setup_storage(USE_GOOGLE_DRIVE, GENESIS_ROOT_OVERRIDE)
LOGGER = EventLogger(PATHS)
HW = detect_hardware()
DEVICE = torch.device(HW["device"])
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False      # dinâmica numérica em FP32 real
CFG = profile_config(PROFILE, HW, MASTER_SEED, MAX_SESSION_HOURS, MAX_GENERATIONS_THIS_SESSION)
print(f"armazenamento: {PATHS.root} ({'Google Drive' if PATHS.on_drive else 'local — NÃO sobrevive à perda da sessão do Colab'})")
print(f"hardware: {HW['gpu_name']} | VRAM {human_bytes(HW['vram_total'])} | CPU x{HW['cpu_count']} | torch {HW['torch']} CUDA {HW['cuda_version']}")
print(f"perfil: {CFG.profile}")
TUNE = autotune(CFG, HW, DEVICE)
print(f"auto-tuning: batch={CFG.batch_size} população={CFG.population_size} mundo={CFG.world_size}²×{CFG.channels_max} "
      f"precisão={CFG.precision} compile={CFG.use_compile}")
print(f"benchmark (ESTIMATIVA): {TUNE['steps_per_sec']:.1f} passos/s · {TUNE['universe_steps_per_sec']:,.0f} universo·passos/s · "
      f"{TUNE['universes_per_sec']:.1f} universos/s · ~{TUNE['estimated_evaluations_per_hour']:,.0f} avaliações/h")
LOGGER.event("boot", hardware=HW, tuning=TUNE, profile=CFG.profile)

# %% [markdown]
# ## 16. TEST SUITE (obrigatória antes da pesquisa)

# %%
if RUN_BOOT_TESTS:
    BOOT_SUITE = run_boot_tests(DEVICE, PATHS.root)
    LOGGER.event("boot_tests_passed", n=len(BOOT_SUITE.results))
else:
    print("⚠ testes de boot desativados pelo usuário (RUN_BOOT_TESTS=False)")

# %% [markdown]
# ## 17. CHECKPOINT SEARCH → NEW GENESIS / RESUME
# A configuração científica (tamanho do mundo, canais, thresholds) fica **travada** pelo checkpoint (research config lock);
# apenas parâmetros de execução (batch, precisão, compile, limite de sessão) seguem o hardware desta sessão.

# %%
RUNTIME_FIELDS = ["batch_size", "precision", "use_compile", "max_session_hours", "max_generations_session",
                  "checkpoint_every_minutes", "dashboard_every_seconds", "population_size"]


def open_lab() -> "GenesisLab":
    probe_ckpt = CheckpointManager(PATHS, LOGGER)
    state, path = (None, "") if FORCE_NEW_GENESIS else probe_ckpt.load_latest()
    if state is not None:
        cfg = GenesisConfig.from_dict(state["config"])
        for f_ in RUNTIME_FIELDS:
            if f_ == "population_size":
                continue
            setattr(cfg, f_, getattr(CFG, f_))
        if cfg.profile != CFG.profile:
            print(f"[resume] perfil do checkpoint ({cfg.profile}) mantido para consistência científica")
        lab = GenesisLab(cfg, HW, PATHS, LOGGER, DEVICE)
        lab.load_state_dict(state)
        lab.counters["sessions"] += 1
        lab.ckpt.last_save_ts = os.path.getmtime(path)
        print(f"RESUME: {path}\n  geração {lab.counters['generation']} · {lab.counters['evaluations']:,} universos avaliados · "
              f"runtime total {fmt_hms(lab.counters['total_runtime'])} · sessão #{lab.counters['sessions']}")
        LOGGER.event("resume", path=path, generation=lab.counters["generation"], evaluations=lab.counters["evaluations"])
    else:
        lab = GenesisLab(CFG, HW, PATHS, LOGGER, DEVICE)
        lab.counters["sessions"] = 1
        print("NEW GENESIS: criando população inicial de universos aleatórios...")
        lab.initialize()
        lab.save_checkpoint("genesis")
    return lab


LAB = open_lab()
print(dashboard_text(LAB, "READY"))

# %% [markdown]
# ## 18. PESQUISA CONTÍNUA
# ```
# while not champion_gate_passed:
#     evolve(); evaluate(); experiment(); archive(); checkpoint()
# ```
# Colab não é infinito: a continuidade é **por checkpoint**. Cada sessão é um trecho da mesma experiência.
# OOM → libera cache, reduz batch, registra, restaura o último checkpoint e continua.

# %%
def resource_check(lab: "GenesisLab"):
    try:
        usage = shutil.disk_usage(lab.paths.root)
        if usage.free < 2e9:
            for f_ in sorted(glob.glob(os.path.join(lab.paths.plots, "*")))[:-50]:
                os.remove(f_)
            lab.logger.event("storage_pressure", echo=True, free=usage.free)
        if HAVE_PSUTIL and psutil.virtual_memory().percent > 90:
            lab.garbage_collect()
            lab.logger.event("ram_pressure", echo=True, percent=psutil.virtual_memory().percent)
    except Exception as e:
        lab.logger.event("resource_check_failed", error=repr(e))


def run_genesis(lab: "GenesisLab") -> str:
    cfg = lab.cfg
    t_end = lab.session_start + cfg.max_session_hours * 3600 - 300
    gens_this_session = 0
    reason = "unknown"
    lab.logger.event("research_start", echo=True, generation=lab.counters["generation"], session=lab.counters["sessions"])
    ref_speed = None
    slow = 0
    dash_n = 0
    try:
        while lab.champion is None:
            if os.path.exists(lab.paths.stop_file):
                reason = "STOP file"; os.remove(lab.paths.stop_file); break
            if time.time() > t_end:
                reason = "MAX_SESSION_HOURS"; break
            if cfg.max_generations_session and gens_this_session >= cfg.max_generations_session:
                reason = "MAX_GENERATIONS_THIS_SESSION"; break
            try:
                row = lab.generation_step()
                gens_this_session += 1
            except Exception as e:
                if not is_oom_error(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                old = cfg.batch_size
                cfg.batch_size = max(2, cfg.batch_size // 2)
                lab.counters["oom_events"] += 1
                lab.logger.event("OOM", echo=True, old_batch=old, new_batch=cfg.batch_size)
                st, p = lab.ckpt.load_latest()
                if st is not None:
                    oom = lab.counters["oom_events"]
                    lab.load_state_dict(st)
                    lab.counters["oom_events"] = oom
                continue
            ups = row["universe_steps_per_s"]
            ref_speed = ups if ref_speed is None else max(ref_speed * 0.99, ups)
            slow = slow + 1 if ups < 0.5 * ref_speed else 0
            if slow >= 3:
                lab.logger.event("performance_degradation", echo=True, speed=ups, reference=ref_speed)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                slow = 0
            if gens_this_session % 10 == 0:
                resource_check(lab)
            if time.time() - lab.last_checkpoint_ts > cfg.checkpoint_every_minutes * 60:
                lab.save_checkpoint("periodic")
            if time.time() - lab.last_dashboard_ts > cfg.dashboard_every_seconds:
                dash_n += 1
                show_dashboard(lab, "SEARCHING", plots=(dash_n % 10 == 0))
                lab.last_dashboard_ts = time.time()
        if lab.champion is not None:
            reason = "GENESIS CHAMPION"
    except KeyboardInterrupt:
        reason = "user interrupt"
        print("\n[interrupt] parada solicitada — salvando estado...")
    except Exception as e:
        reason = f"crash: {e!r}"
        crash = os.path.join(lab.paths.crash, f"crash_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(crash, "w") as f:
            f.write(traceback.format_exc())
        lab.logger.event("crash", echo=True, error=repr(e), report=crash)
        print(traceback.format_exc())
    finally:
        try:
            lab.save_checkpoint(f"session_end:{reason}")
        except Exception as e:
            print(f"[checkpoint] falha ao salvar no encerramento: {e}")
        rp = session_report(lab, reason)
        show_dashboard(lab, "CHAMPION FOUND" if lab.champion else f"PAUSED ({reason})", plots=False)
        print(f"\nrelatório da sessão: {rp}")
        lab.logger.event("research_stop", echo=True, reason=reason, generation=lab.counters["generation"])
    return reason


if START_RESEARCH:
    STOP_REASON = run_genesis(LAB)
else:
    print("START_RESEARCH=False — pesquisa não iniciada (use run_genesis(LAB)).")

# %% [markdown]
# ## 19. Observação dos resultados (somente leitura)
# MAP-Elites, convergência, melhor candidato, zoológico. Ferramentas: `inspect_candidate()`, `compare_candidates(a, b)`,
# `replay_candidate()`, `reproduce_discovery(id)`, `export_candidate()`, `import_candidate(path)`, `profile_genesis()`,
# `results_dataframe()`, `list_discoveries()`.

# %%
plot_map_elites(LAB.map, os.path.join(PATHS.plots, "map_elites.png"), show=True)
plot_convergence(LAB.history, os.path.join(PATHS.plots, "convergence.png"), show=True)
print("Zoo:", LAB.zoo.counts())
print("Bandit de operadores (recompensa média):", LAB.bandit.probs())
_best = LAB.best_candidate()
if _best is not None:
    inspect_candidate(_best, show=True)
if LAB.champion is None:
    print("\nNenhum GENESIS CHAMPION ainda. Isso não é erro: rode novamente (Run all) para continuar a mesma pesquisa.")
