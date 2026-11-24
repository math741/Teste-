# %% [markdown]
# ## ▶ GENESIS — célula principal (boot → testes → autotune → restaurar/criar → observatório 3D → pesquisa ao vivo)
# Esta é a única célula de execução contínua. Ela mantém o viewer disponível e roda o laço de pesquisa até o fim da
# sessão do Colab (ou `MAX_SESSION_HOURS`). Para pausar sem perder nada: botão ■ (checkpoint síncrono no fim).

# %%
GENESIS: Dict[str, Any] = {}


def _free_port(start: int) -> int:
    for p in range(start, start + 40):
        with socket.socket() as s_:
            try:
                s_.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("nenhuma porta livre para o servidor")


def genesis_main():
    display(HTML("<div style='font-family:system-ui;letter-spacing:.3em;color:#9fd3ff;font-weight:700;font-size:15px'>"
                 "◎ GENESIS CIVILIZATION Ω∞</div><div style='font-family:system-ui;color:#7f8ba3;font-size:12px'>"
                 "universo artificial · desenvolvimento · corpos · mentes · afeto funcional · sociedade · cultura · civilização</div>"))
    hw = detect_hardware()
    profile = choose_profile(PROFILE, hw)
    device = torch.device(hw["device"])
    torch.set_flush_denormal(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False          # FP32 é a referência científica
        torch.backends.cudnn.allow_tf32 = False
    if SCIENTIFIC_REPLAY:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type != "cuda":
        print("⚠️ GPU não detectada: rodando em modo DEBUG na CPU. A pesquisa completa requer GPU "
              "(Ambiente de execução → Alterar tipo → GPU).")
    paths = setup_storage(USE_GOOGLE_DRIVE, GENESIS_ROOT_OVERRIDE, GENESIS_WORK_OVERRIDE)
    logger = EventLogger(paths)
    cfg = profile_config(profile, hw, MASTER_SEED)
    cfg.scientific_replay, cfg.human_constraint_experiment = SCIENTIFIC_REPLAY, HUMAN_CONSTRAINT_EXPERIMENT
    cfg.stop_on_champion, cfg.max_session_hours = STOP_ON_CHAMPION, MAX_SESSION_HOURS
    cfg.viewer_port = _free_port(VIEWER_PORT)
    print(f"hardware: {hw.get('gpu_name')} · VRAM {human_bytes(hw.get('vram_total', 0))} · RAM {human_bytes(hw.get('ram_total', 0))} · "
          f"CPU {hw.get('cpu_count')} · torch {hw.get('torch')} · CUDA {hw.get('cuda')} · perfil {profile}")
    print(f"armazenamento: {paths.root} ({'Google Drive' if paths.on_drive else 'local — não persiste se a sessão cair'})")
    boot: Dict[str, Any] = {"profile": profile}
    if RUN_BOOT_TESTS:
        ok, res = run_boot_tests(cfg, paths, device, logger)
        boot.update({"tests_passed": sum(r["ok"] for r in res), "tests_total": len(res)})
        critical = [r for r in res if not r["ok"] and r["test"] != "test_server_protocol"]
        if critical:
            print("❌ GENESIS BOOT FAILED — testes científicos críticos falharam; pesquisa não iniciada:")
            for r in critical:
                print(f"   {r['test']}: {r.get('error')}")
            GENESIS.update({"boot": boot, "tests": res, "paths": paths})
            return GENESIS
    if AUTO_TUNE:
        cfg, tune = benchmark_and_tune(cfg, device, logger, enabled=device.type == "cuda")
        boot["autotune"] = tune
        with open(os.path.join(paths.reports, "autotune.json"), "w") as f:
            json.dump(to_jsonable(tune), f, indent=1)
        print(f"autotune: {tune['final']}")
    uni, ck, info = boot_universe(cfg, paths, device, logger, hw, FORCE_NEW, IMPORT_LEGACY_GENESIS)
    boot["restore"] = info
    print(f"mundo: {'RETOMADO' if info['mode'] == 'resumed' else 'CRIADO'} · tick {uni.tick_count:,} · população {uni.pool.n_alive():,} · "
          f"sessão nº {uni.sessions}" + (f" · checkpoint {os.path.basename(info.get('loaded', ''))}" if info.get("loaded") else ""))
    if info.get("legacy"):
        print(f"Genesis Ω (2D) importado como arquivo separado: {info['legacy']['discoveries']} descobertas · melhor nível "
              f"{info['legacy']['best_level']}")
    desk = ObserverDesk()
    tel = TelemetryEngine(uni, desk)
    tel.state_label = "RESTORING"
    tel.pump(force=True)
    links: Dict[str, Any] = {}
    srv = tun = None
    if START_3D_VIEWER:
        try:
            viewer = Genesis3DViewer(paths, logger)
            static = viewer.install()
            srv = RealtimeServer(desk, lambda: uni, paths, cfg.viewer_port, REMOTE_READ_ONLY, logger, static)
            if srv.start():
                remote = None
                if ENABLE_REMOTE_VIEW:
                    tun = TunnelManager(paths, logger)
                    remote = tun.start(cfg.viewer_port)
                    if remote is None:
                        print(f"túnel remoto indisponível ({tun.error}); o viewer local continua funcionando.")
                links = show_world_links(cfg.viewer_port, srv.view_token, srv.admin_token, remote, REMOTE_READ_ONLY)
                print(f"viewer: three.js {'local' if viewer.vendor_ok else 'via CDN fixado'} · porta {cfg.viewer_port}")
            else:
                print(f"⚠️ viewer não iniciou ({srv.error}); a pesquisa científica continua.")
        except Exception as e:
            print(f"⚠️ viewer falhou ({e!r}); a pesquisa científica continua.")
            logger.event("viewer_failed", error=repr(e), trace=traceback.format_exc()[-1500:])
    GENESIS.update({"uni": uni, "ck": ck, "tel": tel, "desk": desk, "server": srv, "tunnel": tun, "links": links,
                    "cfg": cfg, "paths": paths, "hw": hw, "boot": boot, "logger": logger})
    if START_RESEARCH:
        live = LiveResearch(uni, ck, tel, desk, cfg, paths, logger, links)
        GENESIS["live"] = live
        reason = live.run(MAX_SESSION_HOURS, MAX_TICKS_THIS_SESSION)
        rp = generate_report(uni, hw, boot, reason)
        GENESIS["report"] = rp
        print(f"sessão encerrada ({reason}) · checkpoint {ck.last_info.get('ts')} · relatório {rp}")
        print("Na próxima sessão: Ambiente de execução → Executar tudo — o MESMO mundo é retomado.")
    return GENESIS


genesis_main()
