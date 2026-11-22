# %% [markdown]
# ## 10. Visualização (somente observacional — nunca participa do fitness) e relatórios
# Projeções: PCA dos canais → RGB, seleção de canais, mapa de magnitude. Animações GIF somente para descobertas importantes.

# %%
def rgb_projection(U: np.ndarray, mode: str = "pca", channels: Tuple[int, int, int] = (0, 1, 2),
                   basis: Optional[np.ndarray] = None) -> np.ndarray:
    """U [C,S,S] -> RGB [S,S,3] em [0,1]. Com C>3, três canais nunca são tratados como representação total."""
    C, S, _ = U.shape
    if mode == "magnitude":
        m = np.sqrt((U ** 2).sum(0)); m = m / (m.max() + 1e-9)
        return plt.cm.magma(m)[..., :3]
    if mode == "channels":
        ch = [c if c < C else 0 for c in channels]
        X = np.stack([U[c] for c in ch], -1)
    else:
        Xf = U.reshape(C, -1).T
        Xf = Xf - Xf.mean(0, keepdims=True)
        if basis is None:
            try:
                _, _, Vt = np.linalg.svd(Xf, full_matrices=False)
                basis = Vt[:3]
            except Exception:
                basis = np.eye(C)[:3]
        if basis.shape[0] < 3:
            basis = np.vstack([basis, np.zeros((3 - basis.shape[0], C))])
        X = (Xf @ basis.T).reshape(S, S, 3)
    lo, hi = np.percentile(X, 1), np.percentile(X, 99)
    return np.clip((X - lo) / (hi - lo + 1e-9), 0, 1)


def simulate_frames(lab: "GenesisLab", genome: Genome, env: Optional[Environment] = None, steps: int = 200,
                    every: int = 4, seed: int = 0, S: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    env = env or lab.envs[0]
    gb = lab.sim.prepare([genome], S or lab.cfg.world_size)
    st = lab.sim.init_state(gb, [env], [derive_seed(seed, genome.genome_id)], run_seed=seed)
    init = st.U[0].cpu().numpy()
    frames = []
    for t in range(steps):
        st = lab.sim.step(gb, st)
        if t % every == 0:
            frames.append(st.U[0].cpu().numpy())
    return frames, init


def save_animation(frames: List[np.ndarray], path: str, mode: str = "pca", fps: int = 12) -> Optional[str]:
    try:
        from PIL import Image
        Xall = np.concatenate([f.reshape(f.shape[0], -1).T for f in frames[:: max(1, len(frames) // 10)]])
        Xall = Xall - Xall.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xall, full_matrices=False)
        imgs = []
        for f in frames:
            rgb = (rgb_projection(f, mode, basis=Vt[:3]) * 255).astype(np.uint8)
            im = Image.fromarray(rgb).resize((rgb.shape[1] * 4, rgb.shape[0] * 4), Image.NEAREST)
            imgs.append(im)
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return path
    except Exception as e:
        print(f"[viz] animação não gerada: {e}")
        return None


def _finish(fig, path: Optional[str], show: bool):
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    if show and IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def plot_frames(frames: List[np.ndarray], title: str = "", path: Optional[str] = None, show: bool = True, n: int = 6):
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    fig, ax = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6))
    ax = np.atleast_2d(ax)
    for j, i in enumerate(idx):
        ax[0, j].imshow(rgb_projection(frames[i], "pca")); ax[0, j].set_title(f"t≈{i}", fontsize=8)
        ax[1, j].imshow(rgb_projection(frames[i], "magnitude"))
        ax[0, j].axis("off"); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("PCA→RGB"); ax[1, 0].set_ylabel("|U|")
    fig.suptitle(title, fontsize=10)
    _finish(fig, path, show)


def plot_causal_map(res: TestResult, path: Optional[str] = None, show: bool = True):
    maps, times = res.details.get("causal_map", []), res.details.get("map_times", [])
    if not maps:
        return
    fig, ax = plt.subplots(1, len(maps), figsize=(2 * len(maps), 2.3))
    ax = np.atleast_1d(ax)
    vmax = max(float(np.max(np.asarray(m, np.float32))) for m in maps) + 1e-12
    for a, m, t in zip(ax, maps, times):
        a.imshow(np.log10(np.asarray(m, np.float32) + vmax * 1e-4), cmap="inferno"); a.set_title(f"t={t}", fontsize=8); a.axis("off")
    fig.suptitle(f"Mapa causal log10|U_A-U_B| — {res.candidate_id} ({res.status})", fontsize=9)
    _finish(fig, path, show)


def plot_memory_curve(res: TestResult, path: Optional[str] = None, show: bool = True):
    cv = res.details.get("curve", [])
    if not cv:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    T = [c["T"] for c in cv]
    ax.plot(T, [c["acc"] for c in cv], "o-", label="história (ridge CV)")
    ax.plot(T, [c["null_acc"] for c in cv], "s--", label="controle nulo")
    ax.scatter([c["T"] for c in cv if c["sig"]], [c["acc"] for c in cv if c["sig"]], s=120, facecolors="none",
               edgecolors="g", label="significativo (Holm)")
    ax.axhline(0.5, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("atraso T"); ax.set_ylabel("acurácia balanceada"); ax.set_title(f"M(T) — {res.candidate_id}", fontsize=9)
    ax.legend(fontsize=7)
    _finish(fig, path, show)


def plot_damage_curve(res: TestResult, path: Optional[str] = None, show: bool = True):
    cv = res.details.get("curve", [])
    if not cv:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    x = [c["damage"] * 100 for c in cv]
    ax.plot(x, [c["sim_immediate"] for c in cv], "o--", label="logo após dano")
    ax.plot(x, [c["sim_recovered"] for c in cv], "o-", label="após recuperação")
    ax.plot(x, [c["recovery"] for c in cv], "s-", label="Recovery (relativa)")
    ax.set_xlabel("dano (%)"); ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=7)
    ax.set_title(f"Recovery(damage%) — {res.candidate_id}", fontsize=9)
    _finish(fig, path, show)


def plot_robustness_curve(res: TestResult, path: Optional[str] = None, show: bool = True):
    cv = res.details.get("noise_curve", [])
    conds = res.details.get("conditions", [])
    if not conds:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([c["noise"] for c in cv], [c["similarity"] for c in cv], "o-")
    ax[0].set_xlabel("ruído gaussiano"); ax[0].set_ylabel("similaridade comportamental"); ax[0].set_ylim(0, 1.05)
    labels = [f"{c['condition']}\n{c['level']}" for c in conds]
    ax[1].bar(range(len(conds)), [c["similarity"] for c in conds], color=["tab:green" if c["ok"] else "tab:red" for c in conds])
    ax[1].set_xticks(range(len(conds))); ax[1].set_xticklabels(labels, rotation=70, fontsize=6); ax[1].set_ylim(0, 1.05)
    fig.suptitle(f"Robustez — {res.candidate_id} (score={res.score:.2f})", fontsize=9)
    _finish(fig, path, show)


def plot_map_elites(me: MapElites, path: Optional[str] = None, show: bool = True):
    H = me.heat()
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(H.T, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xlabel(me.axes[0]); ax.set_ylabel(me.axes[1])
    ax.set_title(f"MAP-Elites cobertura={me.coverage():.1%} (máx sobre {me.axes[2] if len(me.axes) > 2 else '-'})", fontsize=8)
    fig.colorbar(im, ax=ax, label="qualidade")
    _finish(fig, path, show)


def plot_convergence(history: List[Dict[str, Any]], path: Optional[str] = None, show: bool = True):
    if len(history) < 2:
        return
    g = [h["generation"] for h in history]
    fig, ax = plt.subplots(1, 4, figsize=(15, 2.8))
    ax[0].plot(g, [h["best_quality"] for h in history]); ax[0].set_title("melhor qualidade", fontsize=8)
    ax[1].plot(g, [h["map_coverage"] for h in history]); ax[1].set_title("cobertura MAP", fontsize=8)
    ax[2].plot(g, [h["species"] for h in history]); ax[2].set_title("espécies", fontsize=8)
    ax[3].plot(g, [h["best_level"] for h in history]); ax[3].set_title("melhor DiscoveryLevel", fontsize=8)
    for a in ax:
        a.set_xlabel("geração", fontsize=7)
    _finish(fig, path, show)
