# %% [markdown]
# ## 18 — REALTIME SERVER (FastAPI + WebSocket) · segurança · túnel HTTPS · embed no Colab
# Segurança: servidor escuta só em `127.0.0.1` (o túnel e o proxy do Colab conectam localmente) · token de
# visualização **aleatório por sessão** (`secrets.token_urlsafe`) vai no *fragmento* da URL (#…, nunca enviado em
# requisições HTTP nem gravado em logs) · token **admin separado**, nunca no link remoto; com `REMOTE_READ_ONLY`
# o admin é recusado em qualquer requisição que chegue pelo túnel · rate limit por IP (admin muito mais restrito) ·
# validação de entrada em todo endpoint · **sem endpoint de shell, sem eval, sem pickle** (JSON/msgpack de tipos
# primitivos apenas) · arquivos estáticos por lista branca (sem path traversal) · documentação automática desligada.

# %%
HAVE_WEB = HAVE_MSGPACK and all([ensure_package("fastapi", "fastapi>=0.110,<1"), ensure_package("uvicorn", "uvicorn>=0.29,<1"),
                                  ensure_package("websockets", "websockets>=10,<16")])
if HAVE_WEB:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
    from fastapi.responses import Response, JSONResponse, RedirectResponse, HTMLResponse


class RateLimiter:
    def __init__(self, per_minute: int):
        self.rate, self.cap = per_minute / 60.0, float(per_minute)
        self.b: Dict[str, Tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            tok, last = self.b.get(key, (self.cap, now))
            tok = min(self.cap, tok + (now - last) * self.rate)
            if tok < 1:
                self.b[key] = (tok, now)
                return False
            self.b[key] = (tok - 1, now)
            if len(self.b) > 10000:
                self.b = dict(list(self.b.items())[-5000:])
            return True


PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "ngrok-trace-id", "x-real-ip")
VENDOR_FILES = {"three.module.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js",
                "VRButton.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/webxr/VRButton.js"}
STATIC_FILES = {"app.js": "application/javascript", "style.css": "text/css", "msgpack.js": "application/javascript"}


class RealtimeServer:
    def __init__(self, desk: ObserverDesk, uni_ref: Callable[[], "UniverseEngine"], paths: CivPaths, port: int,
                 read_only: bool, logger: Optional[EventLogger], static_dir: str):
        self.desk, self.uni_ref, self.paths, self.port, self.read_only, self.logger = desk, uni_ref, paths, port, read_only, logger
        self.static_dir = static_dir
        self.view_token = secrets.token_urlsafe(18)
        self.admin_token = secrets.token_urlsafe(24)
        self.rl_api, self.rl_admin, self.rl_ws = RateLimiter(600), RateLimiter(12), RateLimiter(30)
        self.clients = 0
        self.thread: Optional[threading.Thread] = None
        self.server = None
        self.started = False
        self.error: Optional[str] = None
        self.app = self._make_app() if HAVE_WEB else None

    # ------------------------------------------------------------------ auth
    @staticmethod
    def _via_tunnel(headers) -> bool:
        """Requisição chegou pelo túnel público? (Cloudflare/ngrok). O proxy autenticado do Colab também pode mandar
        X-Forwarded-For, mas serve só o navegador do próprio usuário — não é tratado como acesso remoto."""
        host = headers.get("host", "").split(":")[0].lower()
        if any(h in headers for h in ("cf-connecting-ip", "cf-ray", "cf-ipcountry")) or any(k.lower().startswith("ngrok") for k in headers.keys()):
            return True
        if host.endswith((".trycloudflare.com", ".ngrok-free.app", ".ngrok.app", ".ngrok.io", ".ngrok.dev", ".ngrok-free.dev")):
            return True
        if host.endswith((".googleusercontent.com", ".colab.dev")) or host in ("127.0.0.1", "localhost"):
            return False
        return any(h in headers for h in PROXY_HEADERS)

    def _token_ok(self, tok: Optional[str], admin: bool = False) -> bool:
        if not tok or not isinstance(tok, str) or len(tok) > 64:
            return False
        ref = self.admin_token if admin else self.view_token
        return secrets.compare_digest(tok.encode(), ref.encode()) or (not admin and secrets.compare_digest(tok.encode(), self.admin_token.encode()))

    def _check(self, request: "Request", admin: bool = False):
        ip = (request.client.host if request.client else "?") + "|" + request.headers.get("cf-connecting-ip", "")
        if not (self.rl_admin if admin else self.rl_api).allow(ip):
            raise HTTPException(429, "rate limit")
        tok = request.headers.get("x-genesis-token")
        if admin:
            if self.read_only and self._via_tunnel(request.headers):
                raise HTTPException(403, "admin desabilitado no acesso remoto (REMOTE_READ_ONLY=True)")
            if not self._token_ok(request.headers.get("x-genesis-admin"), admin=True):
                raise HTTPException(401, "admin token inválido")
        elif not self._token_ok(tok):
            raise HTTPException(401, "token inválido")

    # ------------------------------------------------------------------ app
    def _make_app(self):
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        srv = self

        imap = Genesis3DViewer.importmap_hash(srv.static_dir) if os.path.exists(os.path.join(srv.static_dir, "index.html")) else None
        script_src = "'self' https://cdn.jsdelivr.net https://unpkg.com" + (f" '{imap}'" if imap else "")

        @app.middleware("http")
        async def security_headers(request: Request, call_next):
            resp = await call_next(request)
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["Referrer-Policy"] = "no-referrer"
            resp.headers["Content-Security-Policy"] = (
                f"default-src 'self'; script-src {script_src}; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; connect-src 'self' ws: wss:; worker-src 'self' blob:; "
                "frame-ancestors 'self' https://colab.research.google.com https://*.googleusercontent.com")
            return resp

        @app.get("/health")
        async def health():
            return {"ok": True, "version": GENESIS_VERSION, "proto": LIVE_PROTOCOL, "online": srv.desk.online,
                    "ts": iso_now()}

        @app.get("/")
        async def index():
            p = os.path.join(srv.static_dir, "index.html")
            html = open(p, encoding="utf-8").read().replace("__VERSION__", GENESIS_VERSION + "-" + ENGINE_VERSIONS_HASH[:8])
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})

        @app.get("/static/{name}")
        async def static(name: str):
            if name not in STATIC_FILES:
                raise HTTPException(404)
            p = os.path.join(srv.static_dir, name)
            return Response(open(p, "rb").read(), media_type=STATIC_FILES[name], headers={"Cache-Control": "no-cache"})

        @app.get("/vendor/{name}")
        async def vendor(name: str):
            if name not in VENDOR_FILES:
                raise HTTPException(404)
            p = os.path.join(srv.static_dir, "vendor", name)
            if os.path.exists(p):
                return Response(open(p, "rb").read(), media_type="application/javascript",
                                headers={"Cache-Control": "public, max-age=86400"})
            return RedirectResponse(VENDOR_FILES[name])          # fallback: CDN fixado na mesma versão

        def ask(kind, args, timeout=4.0):
            r = srv.desk.ask(kind, args, timeout)
            if isinstance(r, dict) and r.get("error") in ("busy", "timeout"):
                raise HTTPException(503, r.get("note", r["error"]))
            return r

        def mp(obj):
            return Response(obj if isinstance(obj, bytes) else pack_msg(obj), media_type="application/msgpack")

        @app.get("/api/status")
        async def status(request: Request):
            srv._check(request)
            with srv.desk.lock:
                return JSONResponse(to_jsonable(srv.desk.status))

        @app.get("/api/world")
        async def world(request: Request):
            srv._check(request)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "world", {})))

        @app.get("/api/agent/{aid}")
        async def agent(aid: int, request: Request):
            srv._check(request)
            if not 0 < aid < 2 ** 53:
                raise HTTPException(422, "aid fora do intervalo")
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "agent", {"aid": aid})))

        @app.get("/api/city/{sid}")
        async def city(sid: int, request: Request):
            srv._check(request)
            if not 0 < sid < 10 ** 9:
                raise HTTPException(422)
            r = await asyncio.to_thread(ask, "city", {"sid": sid})
            if r is None:
                raise HTTPException(404)
            return JSONResponse(to_jsonable(r))

        @app.get("/api/develop/{aid}")
        async def develop(aid: int, request: Request):
            srv._check(request)
            if not 0 < aid < 2 ** 53:
                raise HTTPException(422)
            r = await asyncio.to_thread(ask, "develop", {"aid": aid})
            if r is None:
                raise HTTPException(404)
            return mp(r)

        @app.get("/api/history")
        async def history(request: Request, since: int = 0, limit: int = 300):
            srv._check(request)
            if not (0 <= since < 2 ** 53 and 1 <= limit <= 2000):
                raise HTTPException(422)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "history", {"since": since, "limit": limit})))

        @app.get("/api/discoveries")
        async def discoveries(request: Request):
            srv._check(request)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "discoveries", {})))

        @app.get("/api/ladders")
        async def ladders(request: Request):
            srv._check(request)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "ladders", {})))

        @app.get("/api/replay/{did}")
        async def replay(did: str, request: Request):
            srv._check(request)
            if not re.fullmatch(r"D\d{5}-[A-Z0-9]{1,4}-[0-9a-f]{6}", did):
                raise HTTPException(422, "id inválido")
            d = await asyncio.to_thread(srv.uni_ref().archive.load_discovery, did)
            if d is None or "replay" not in d:
                raise HTTPException(404, "sem replay")
            rp = d["replay"]
            return mp({"t": "replay", "id": did, "level": d.get("level"), "name": d.get("level_name"), "size": rp["size"],
                       "ticks": rp["ticks"], "k": int(rp["pos"].shape[1]), "pos": rp["pos"].astype(np.float32).tobytes(),
                       "arena_of": rp["arena_of"].astype(np.int16).tobytes(), "layout": rp["layout"]})

        @app.get("/api/ring")
        async def ring_index(request: Request):
            srv._check(request)
            with srv.desk.lock:
                return JSONResponse(list(srv.desk.ring_index))

        @app.get("/api/ring/{i}")
        async def ring(i: int, request: Request):
            srv._check(request)
            if not 0 <= i < 100000:
                raise HTTPException(422)
            r = await asyncio.to_thread(ask, "ring", {"i": i})
            if r is None:
                raise HTTPException(404)
            return mp(r)

        @app.get("/api/search")
        async def search(request: Request, q: str = ""):
            srv._check(request)
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", q):
                raise HTTPException(422)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "search", {"q": q})))

        @app.get("/api/lineage/{aid}")
        async def lineage(aid: int, request: Request):
            srv._check(request)
            if not 0 < aid < 2 ** 53:
                raise HTTPException(422)
            return JSONResponse(to_jsonable(await asyncio.to_thread(ask, "lineage", {"aid": aid})))

        def user_file(name):
            return os.path.join(srv.paths.viewer_user, f"{name}.json")

        for name in ("favorites", "notes"):
            def make(name=name):
                @app.get(f"/api/{name}")
                async def get_user(request: Request):
                    srv._check(request)
                    p = user_file(name)
                    return JSONResponse(json.load(open(p)) if os.path.exists(p) else [])

                @app.post(f"/api/{name}")
                async def post_user(request: Request):
                    srv._check(request)
                    body = await request.body()
                    if len(body) > 65536:
                        raise HTTPException(413)
                    try:
                        data = json.loads(body)
                    except Exception:
                        raise HTTPException(422, "JSON inválido")
                    if not isinstance(data, list) or len(data) > 500:
                        raise HTTPException(422, "lista de até 500 itens")
                    clean = []
                    for it in data:
                        if not isinstance(it, dict):
                            raise HTTPException(422)
                        c = {}
                        for k, v in it.items():
                            if not isinstance(k, str) or len(k) > 32 or k not in ("type", "id", "label", "text", "cam", "tick", "ts"):
                                raise HTTPException(422, "campo não permitido")
                            if isinstance(v, str):
                                c[k] = v[:2000]
                            elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                                c[k] = v
                            elif isinstance(v, list) and len(v) <= 9 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
                                c[k] = v
                            else:
                                raise HTTPException(422, "valor não permitido")
                        clean.append(c)
                    with open(user_file(name), "w") as f:
                        json.dump(clean, f)
                    return {"ok": True, "n": len(clean), "note": "observações do usuário nunca contam como evidência"}
            make()

        # ------------------------------------------------------------------ admin (segundo token, rate limit, local)
        ADMIN = {"pause", "resume", "checkpoint", "focus", "fork", "dsl", "unfocus"}

        @app.post("/api/admin/{action}")
        async def admin(action: str, request: Request):
            srv._check(request, admin=True)
            if action not in ADMIN:
                raise HTTPException(404)
            body = await request.body()
            if len(body) > 16384:
                raise HTTPException(413)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                raise HTTPException(422, "JSON inválido")
            if not isinstance(data, dict):
                raise HTTPException(422)
            if action == "focus":
                aid = data.get("aid")
                if not isinstance(aid, int) or isinstance(aid, bool) or not 0 < aid < 2 ** 53:
                    raise HTTPException(422)
                data = {"aid": aid}
            elif action == "fork":
                iv, ticks = data.get("intervention"), data.get("ticks", 500)
                if not isinstance(iv, dict) or iv.get("kind") not in FORK_INTERVENTIONS or not isinstance(ticks, int):
                    raise HTTPException(422)
                if iv.get("kind") == "block_action" and iv.get("channel") not in ACTION_NAMES:
                    raise HTTPException(422)
                data = {"intervention": {k: v for k, v in iv.items() if k in ("kind", "channel", "value")
                                         and isinstance(v, (str, int, float)) and not isinstance(v, bool)},
                        "ticks": int(np.clip(ticks, 25, 5000))}
            elif action == "dsl":
                try:
                    data = {"program": validate_dsl(data.get("program"))}
                except ValueError as e:
                    raise HTTPException(422, f"DSL inválida: {e}")
            else:
                data = {}
            try:
                srv.desk.admin_queue.put_nowait({"action": action, "data": data, "ts": iso_now(),
                                                 "remote": srv._via_tunnel(request.headers)})
            except queue.Full:
                raise HTTPException(503, "fila admin cheia")
            if srv.logger:
                srv.logger.event("admin_request", action=action)
            return {"ok": True, "queued": action}

        # ------------------------------------------------------------------ WebSocket ao vivo
        @app.websocket("/ws/live")
        async def live(ws: WebSocket):
            ip = ws.client.host if ws.client else "?"
            tok = ws.query_params.get("t")
            if not srv.rl_ws.allow(ip) or not srv._token_ok(tok):
                await ws.close(code=4401)
                return
            await ws.accept()
            cid = uuid.uuid4().hex[:8]
            srv.clients += 1
            state = {"seq": 0, "base_seq": -1, "status_t": 0.0, "body_t": 0.0, "heat_t": 0.0, "lab_t": 0.0, "lab": False,
                     "alive": True, "max_hz": 10.0}

            async def receiver():
                while state["alive"]:
                    try:
                        msg = await ws.receive_text()
                    except Exception:
                        state["alive"] = False
                        return
                    if len(msg) > 2048:
                        continue
                    try:
                        m = json.loads(msg)
                    except Exception:
                        continue
                    if not isinstance(m, dict):
                        continue
                    t = m.get("t")
                    if t == "select":
                        a = m.get("aid")
                        if isinstance(a, int) and not isinstance(a, bool) and 0 < a < 2 ** 53:
                            srv.desk.selected[cid] = a
                        elif a is None:
                            srv.desk.selected.pop(cid, None)
                    elif t == "view":
                        try:
                            v = tuple(float(m[k]) for k in ("x0", "y0", "x1", "y1"))
                            n = int(np.clip(int(m.get("max_agents", 4000)), 50, 50000))
                            if all(math.isfinite(x) for x in v):
                                srv.desk.views[cid] = (*v, n)
                            state["max_hz"] = float(np.clip(float(m.get("hz", 10)), 0.5, 30))
                        except Exception:
                            pass
                    elif t == "layer":
                        lay = m.get("name")
                        if lay in HEAT_LAYERS:
                            srv.desk.layers[cid] = lay
                        else:
                            srv.desk.layers.pop(cid, None)
                    elif t == "lab":
                        state["lab"] = bool(m.get("on"))
                    elif t == "resync":
                        state["base_seq"] = -1
                    elif t == "ping":
                        ts = m.get("ts")
                        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
                            try:
                                await ws.send_text(json.dumps({"t": "pong", "ts": ts, "server": time.time()}))
                            except Exception:
                                state["alive"] = False
            rt = asyncio.create_task(receiver())
            try:
                await ws.send_text(json.dumps({"t": "hello", "proto": LIVE_PROTOCOL, "version": GENESIS_VERSION,
                                               "read_only": srv.read_only, "server": time.time(),
                                               "claim": "universo artificial; afeto funcional ≠ experiência subjetiva"}))
                last_send = 0.0
                while state["alive"]:
                    now = time.time()
                    if srv.desk.base is not None and state["base_seq"] != srv.desk.base_seq:
                        await ws.send_bytes(srv.desk.base); state["base_seq"] = srv.desk.base_seq
                    lt = srv.desk.latest()
                    if lt is not None and lt[0] != state["seq"] and now - last_send >= 1.0 / state["max_hz"]:
                        # backpressure: sempre o frame MAIS RECENTE; frames antigos são descartados
                        await ws.send_bytes(lt[1]); state["seq"] = lt[0]; last_send = now
                    if now - state["status_t"] >= 1.0:
                        with srv.desk.lock:
                            st = dict(srv.desk.status)
                        st["clients"] = srv.clients
                        await ws.send_text(json.dumps(to_jsonable(st)))
                        state["status_t"] = now
                    a = srv.desk.selected.get(cid)
                    if a is not None and now - state["body_t"] >= 0.25 and a in srv.desk.body:
                        await ws.send_bytes(srv.desk.body[a]); state["body_t"] = now
                    lay = srv.desk.layers.get(cid)
                    if lay and now - state["heat_t"] >= 2.0 and lay in srv.desk.heat:
                        await ws.send_bytes(srv.desk.heat[lay]); state["heat_t"] = now
                    if state["lab"] and srv.desk.lab is not None and now - state["lab_t"] >= 0.5:
                        await ws.send_bytes(srv.desk.lab); state["lab_t"] = now
                    await asyncio.sleep(0.03)
            except (WebSocketDisconnect, RuntimeError):
                pass
            except Exception as e:
                if srv.logger:
                    srv.logger.event("ws_error", error=repr(e))
            finally:
                state["alive"] = False
                rt.cancel()
                srv.clients -= 1
                for d_ in (srv.desk.selected, srv.desk.views, srv.desk.layers):
                    d_.pop(cid, None)

        return app

    # ------------------------------------------------------------------ ciclo de vida
    def start(self) -> bool:
        if not HAVE_WEB:
            self.error = "fastapi/uvicorn/msgpack indisponíveis"
            return False
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False,
                             ws_max_size=1 << 20, lifespan="off")
        self.server = uvicorn.Server(cfg)

        def run():
            try:
                self.server.run()
            except Exception as e:
                self.error = repr(e)
        self.thread = threading.Thread(target=run, name="genesis-server", daemon=True)
        self.thread.start()
        for _ in range(100):
            if getattr(self.server, "started", False):
                self.started = True
                return True
            if self.error:
                return False
            time.sleep(0.1)
        self.error = self.error or "timeout ao iniciar"
        return False

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


class TunnelManager:
    """Adaptador de túnel HTTPS: Cloudflare quick tunnel (sem conta) ou ngrok (se NGROK_AUTHTOKEN estiver em variável de
    ambiente ou em Colab Secret). Segredos nunca são gravados no notebook nem passados por linha de comando."""
    CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    NGROK = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"

    def __init__(self, paths: CivPaths, logger: Optional[EventLogger]):
        self.paths, self.logger = paths, logger
        self.proc: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None
        self.provider: Optional[str] = None
        self.error: Optional[str] = None

    @staticmethod
    def _secret(name: str) -> Optional[str]:
        v = os.environ.get(name)
        if v:
            return v
        if IN_COLAB:
            try:
                from google.colab import userdata
                return userdata.get(name)
            except Exception:
                return None
        return None

    def _download(self, url: str, dest: str) -> bool:
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            return True
        try:
            import urllib.request
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            self.error = f"download falhou: {e}"
            return False

    def _read_url(self, pattern: str, timeout: float) -> Optional[str]:
        t0 = time.time()
        rx = re.compile(pattern)
        while time.time() - t0 < timeout and self.proc and self.proc.poll() is None:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.1); continue
            m = rx.search(line)
            if m:
                return m.group(0)
        return None

    def _drain(self):
        def run():
            try:
                for _ in iter(self.proc.stdout.readline, ""):
                    pass
            except Exception:
                pass
        threading.Thread(target=run, daemon=True, name="tunnel-drain").start()

    def start(self, port: int, timeout: float = 45.0) -> Optional[str]:
        bindir = os.path.join(self.paths.work, "bin")
        os.makedirs(bindir, exist_ok=True)
        cf = os.path.join(bindir, "cloudflared")
        if self._download(self.CLOUDFLARED, cf):
            os.chmod(cf, 0o755)
            try:
                self.proc = subprocess.Popen([cf, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                url = self._read_url(r"https://[a-z0-9\-]+\.trycloudflare\.com", timeout)
                if url:
                    self.url, self.provider = url, "cloudflare"
                    self._drain()
                    return url
                self.stop()
                self.error = "cloudflared não publicou URL"
            except Exception as e:
                self.error = f"cloudflared: {e}"
        tok = self._secret("NGROK_AUTHTOKEN")
        if tok:
            tgz = os.path.join(bindir, "ngrok.tgz")
            ng = os.path.join(bindir, "ngrok")
            if self._download(self.NGROK, tgz):
                try:
                    import tarfile
                    with tarfile.open(tgz) as t:
                        member = t.getmember("ngrok")
                        member.name = "ngrok"
                        t.extract(member, bindir, filter="data") if hasattr(tarfile, "data_filter") else t.extract(member, bindir)
                    os.chmod(ng, 0o755)
                    env = dict(os.environ, NGROK_AUTHTOKEN=tok)
                    self.proc = subprocess.Popen([ng, "http", str(port), "--log", "stdout", "--log-format", "json"],
                                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                    url = self._read_url(r"https://[a-zA-Z0-9\-\.]+\.ngrok[a-z\-\.]*\.(app|io|dev)", timeout)
                    if url:
                        self.url, self.provider = url, "ngrok"
                        self._drain()
                        return url
                    self.stop()
                    self.error = (self.error or "") + "; ngrok não publicou URL"
                except Exception as e:
                    self.error = f"ngrok: {e}"
        if self.logger:
            self.logger.event("tunnel_unavailable", reason=self.error)
        return None

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
