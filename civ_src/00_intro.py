# %% [markdown]
# # PROJECT GENESIS CIVILIZATION Ω∞
# ### Artificial Universe · Developmental Life · Embodied Minds · Functional Affect · Society · Culture · Civilization · 3D Live World
#
# > **GENESIS CIVILIZATION Ω∞ é um universo artificial, não uma simulação atomística do Universo físico real.**
# > Nada aqui afirma vida real, consciência, sentimentos subjetivos, AGI, humanidade real ou cérebro humano real.
#
# O laboratório separa sempre: **comportamento observado** → **mecanismo funcional demonstrado** (probe com controles,
# replicação e ablação) → **interpretação humana** → *experiência subjetiva* (nunca assumida).
#
# ### Como usar
# 1. `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU` (sem GPU roda em modo DEBUG na CPU).
# 2. `Ambiente de execução → Executar tudo`. Autorize o Google Drive quando pedido (é onde o universo persiste).
# 3. No fim do boot aparece **OPEN GENESIS WORLD**: o observatório 3D (também embutido aqui no Colab) e um link remoto
#    HTTPS autenticado para abrir de qualquer computador ou celular enquanto o Colab estiver rodando.
# 4. A pesquisa roda até o Colab encerrar a sessão (checkpoint atômico a cada poucos minutos). Na próxima vez,
#    `Executar tudo` retoma o **mesmo** mundo: idade, agentes, cultura, assentamentos, fila científica e história.
#    Pausar: botão ■ da célula 23 (salva tudo).
#
# ### Legenda de status usada em todo o laboratório
# `IMPLEMENTED` código existe e roda · `TESTED` coberto por teste com controles · `BENCHMARKED` medido neste hardware ·
# `SIMULATION HYPOTHESIS` pergunta aberta que o sistema permite investigar · `NOT YET VALIDATED` sem evidência ainda.
#
# Regra suprema: **não construir a resposta dentro do universo.** Não existe `build_house()`, `create_civilization()`,
# variável `fear`, `happiness`, linguagem pronta ou fitness de "parecer humano". Estruturas análogas só são rotuladas
# quando um instrumento externo, com controles positivo e negativo, demonstra a função.

# %%
# @title ⚙️ Configuração (defaults fortes — nada precisa ser mudado) { display-mode: "form" }
import os

PROFILE = "AUTO"  # @param ["AUTO", "DEBUG", "FAST", "RESEARCH", "EXTREME", "A100_80GB"]
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
START_RESEARCH = True  # @param {type:"boolean"}
START_3D_VIEWER = True  # @param {type:"boolean"}
ENABLE_REMOTE_VIEW = True  # @param {type:"boolean"}
REMOTE_READ_ONLY = True  # @param {type:"boolean"}
STOP_ON_CHAMPION = False  # @param {type:"boolean"}
AUTO_TUNE = True  # @param {type:"boolean"}
SCIENTIFIC_REPLAY = False  # @param {type:"boolean"}
HUMAN_CONSTRAINT_EXPERIMENT = False  # @param {type:"boolean"}
RUN_BOOT_TESTS = True  # @param {type:"boolean"}
IMPORT_LEGACY_GENESIS = True  # @param {type:"boolean"}
FORCE_NEW = False  # @param {type:"boolean"}
MAX_SESSION_HOURS = 24.0  # @param {type:"number"}
MAX_TICKS_THIS_SESSION = 0  # @param {type:"integer"}
MASTER_SEED = 20261124  # @param {type:"integer"}
VIEWER_PORT = 8765  # @param {type:"integer"}

_env = os.environ.get
PROFILE = _env("GENESIS_PROFILE", PROFILE)
USE_GOOGLE_DRIVE = _env("GENESIS_USE_DRIVE", "1" if USE_GOOGLE_DRIVE else "0") == "1"
START_RESEARCH = _env("GENESIS_START_RESEARCH", "1" if START_RESEARCH else "0") == "1"
START_3D_VIEWER = _env("GENESIS_VIEWER", "1" if START_3D_VIEWER else "0") == "1"
ENABLE_REMOTE_VIEW = _env("GENESIS_REMOTE", "1" if ENABLE_REMOTE_VIEW else "0") == "1"
REMOTE_READ_ONLY = _env("GENESIS_READ_ONLY", "1" if REMOTE_READ_ONLY else "0") == "1"
STOP_ON_CHAMPION = _env("GENESIS_STOP_ON_CHAMPION", "1" if STOP_ON_CHAMPION else "0") == "1"
AUTO_TUNE = _env("GENESIS_AUTO_TUNE", "1" if AUTO_TUNE else "0") == "1"
SCIENTIFIC_REPLAY = _env("GENESIS_SCIENTIFIC_REPLAY", "1" if SCIENTIFIC_REPLAY else "0") == "1"
HUMAN_CONSTRAINT_EXPERIMENT = _env("GENESIS_HUMAN_CONSTRAINT", "1" if HUMAN_CONSTRAINT_EXPERIMENT else "0") == "1"
RUN_BOOT_TESTS = _env("GENESIS_RUN_TESTS", "1" if RUN_BOOT_TESTS else "0") == "1"
IMPORT_LEGACY_GENESIS = _env("GENESIS_IMPORT_LEGACY", "1" if IMPORT_LEGACY_GENESIS else "0") == "1"
FORCE_NEW = _env("GENESIS_FORCE_NEW", "1" if FORCE_NEW else "0") == "1"
MAX_SESSION_HOURS = float(_env("GENESIS_MAX_HOURS", MAX_SESSION_HOURS))
MAX_TICKS_THIS_SESSION = int(_env("GENESIS_MAX_TICKS", MAX_TICKS_THIS_SESSION))
MASTER_SEED = int(_env("GENESIS_SEED", MASTER_SEED))
VIEWER_PORT = int(_env("GENESIS_PORT", VIEWER_PORT))
GENESIS_ROOT_OVERRIDE = _env("GENESIS_ROOT", "")
GENESIS_WORK_OVERRIDE = _env("GENESIS_WORK", "")
