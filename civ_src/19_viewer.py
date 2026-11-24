# %% [markdown]
# ## 19 — GENESIS 3D VIEWER: instalação do observatório (Three.js no cliente)
# O notebook grava os arquivos do viewer (`index.html`, `app.js`, `style.css`, `msgpack.js`) no SSD de trabalho e
# baixa Three.js **fixado em 0.160.0** do registro npm (tarball verificado por SHA-512 do próprio registro). Se o
# download falhar, o servidor redireciona para o CDN na mesma versão. Renderização pesada acontece no navegador do
# usuário (a GPU do Colab fica para a ciência).

# %%
VIEWER_FILES_B64 = "@@VIEWER_FILES@@"
THREE_VERSION = "0.160.0"


class Genesis3DViewer:
    TGZ = f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz"
    META = f"https://registry.npmjs.org/three/{THREE_VERSION}"
    WANT = {"package/build/three.module.js": "three.module.js",
            "package/examples/jsm/controls/OrbitControls.js": "OrbitControls.js",
            "package/examples/jsm/webxr/VRButton.js": "VRButton.js"}

    def __init__(self, paths: CivPaths, logger: Optional[EventLogger]):
        self.paths, self.logger = paths, logger
        self.static_dir = os.path.join(paths.work, "viewer", "static")
        self.vendor_ok = False

    def install(self) -> str:
        if VIEWER_FILES_B64.startswith("@@"):
            raise RuntimeError("arquivos do viewer não embutidos (build incompleto)")
        files = json.loads(base64.b64decode(VIEWER_FILES_B64).decode("utf-8"))
        os.makedirs(os.path.join(self.static_dir, "vendor"), exist_ok=True)
        for name, content in files.items():
            if not re.fullmatch(r"[a-z]+\.(html|js|css)", name):
                continue
            with open(os.path.join(self.static_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        self.vendor_ok = self._vendor()
        return self.static_dir

    def _vendor(self) -> bool:
        vd = os.path.join(self.static_dir, "vendor")
        if all(os.path.exists(os.path.join(vd, v)) for v in self.WANT.values()):
            return True
        try:
            import urllib.request, tarfile
            meta = json.loads(urllib.request.urlopen(self.META, timeout=20).read())
            integrity = meta.get("dist", {}).get("integrity", "")
            dl = os.path.join(self.paths.work, "cache", f"three-{THREE_VERSION}.tgz")
            if not os.path.exists(dl):
                urllib.request.urlretrieve(self.TGZ, dl + ".part")
                os.replace(dl + ".part", dl)
            if integrity.startswith("sha512-"):
                got = base64.b64encode(hashlib.sha512(open(dl, "rb").read()).digest()).decode()
                if got != integrity.split("-", 1)[1]:
                    os.remove(dl)
                    raise RuntimeError("integridade do tarball three.js não confere")
            with tarfile.open(dl) as t:
                for m in t.getmembers():
                    if m.name in self.WANT and m.isfile():
                        data = t.extractfile(m).read()
                        with open(os.path.join(vd, self.WANT[m.name]), "wb") as f:
                            f.write(data)
            return all(os.path.exists(os.path.join(vd, v)) for v in self.WANT.values())
        except Exception as e:
            if self.logger:
                self.logger.event("viewer_vendor_fallback_cdn", error=repr(e))
            return False

    @staticmethod
    def importmap_hash(static_dir: str) -> Optional[str]:
        html = open(os.path.join(static_dir, "index.html"), encoding="utf-8").read()
        m = re.search(r'<script type="importmap">(.*?)</script>', html, flags=re.S)
        if not m:
            return None
        return "sha256-" + base64.b64encode(hashlib.sha256(m.group(1).encode("utf-8")).digest()).decode()


def colab_proxy_url(port: int) -> Optional[str]:
    if not IN_COLAB:
        return None
    try:
        from google.colab.output import eval_js
        return eval_js(f"google.colab.kernel.proxyPort({port})")
    except Exception:
        return None


def show_world_links(port: int, view_token: str, admin_token: str, remote_url: Optional[str], read_only: bool,
                     embed: bool = True, height: int = 760):
    """Botão OPEN GENESIS WORLD + link remoto + embed local. Tokens só no fragmento (#) da URL; o admin nunca vai no
    link remoto. Tokens valem só nesta sessão."""
    local = colab_proxy_url(port)
    local_link = (local.rstrip("/") + f"/#t={view_token}&a={admin_token}") if local else f"http://127.0.0.1:{port}/#t={view_token}&a={admin_token}"
    remote_link = (remote_url.rstrip("/") + f"/#t={view_token}") if remote_url else None
    btn = ("display:inline-block;padding:12px 22px;border-radius:24px;background:linear-gradient(90deg,#2a5cff,#3fd0c9);"
           "color:white;font-weight:700;letter-spacing:.08em;text-decoration:none;font-family:system-ui;margin:4px 8px 4px 0")
    html_ = f'<div style="font-family:system-ui;color:#cfe0ff;background:#070b14;padding:14px 18px;border-radius:14px">'
    html_ += f'<a href="{_html.escape(local_link)}" target="_blank" style="{btn}">🌍 OPEN GENESIS WORLD</a>'
    if remote_link:
        html_ += (f'<a href="{_html.escape(remote_link)}" target="_blank" style="{btn};background:linear-gradient(90deg,#7b4dff,#c04dff)">'
                  f'📱 LINK REMOTO (HTTPS, {"somente leitura" if read_only else "leitura"})</a>')
        html_ += (f'<div style="margin-top:8px;font-size:12px;color:#8fa3c7">Remoto: <code>{_html.escape(remote_link)}</code> — abre em qualquer '
                  f'computador/celular enquanto este Colab estiver rodando. Token por sessão; admin desabilitado no remoto.</div>')
    else:
        html_ += '<div style="margin-top:8px;font-size:12px;color:#ffb65c">Túnel remoto indisponível agora: viewer local funciona normalmente.</div>'
    html_ += '</div>'
    display(HTML(html_))
    if embed and IN_COLAB:
        try:
            from google.colab import output
            output.serve_kernel_port_as_iframe(port, path=f"/#t={view_token}&a={admin_token}", height=height)
        except Exception as e:
            print(f"[viewer] embed indisponível ({e}); use o botão acima.")
    return {"local": local_link, "remote": remote_link}
