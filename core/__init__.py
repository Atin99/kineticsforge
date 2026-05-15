def _optional_import(import_fn):
    try:
        import_fn()
    except ModuleNotFoundError:
        pass


def _load_neural_ode():
    exec("from core.neural_ode import *", globals())


def _load_physics_constraints():
    exec("from core.physics_constraints import *", globals())


def _load_koopman():
    exec("from core.koopman import *", globals())


def _load_sindy():
    exec("from core.sindy_discovery import *", globals())


def _load_lyapunov():
    exec("from core.lyapunov import *", globals())


def _load_uncertainty():
    exec("from core.uncertainty import *", globals())


def _load_utils():
    exec("from core.utils import *", globals())


_optional_import(_load_neural_ode)
_optional_import(_load_physics_constraints)
_optional_import(_load_koopman)
_optional_import(_load_sindy)
_optional_import(_load_lyapunov)
_optional_import(_load_uncertainty)
_optional_import(_load_utils)

from core.india_context import *  # noqa: F401,F403
from core.evidence_registry import *  # noqa: F401,F403
from core.physics_audit import *  # noqa: F401,F403
from core.dimensional_analysis import *  # noqa: F401,F403
