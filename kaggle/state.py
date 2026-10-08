"""Constants and helpers shared by the modules of server.py.

A Kaggle script kernel runs a single file, so `kis` pastes this file over each
`from state import ...` line when it pushes the script (see kis/kaggle.py). All the
modules then share one namespace: keep top-level names unique across kaggle/, and
import with `from state import a, b` only.

Nothing here depends on the session's settings. Those derive from the CONFIG that
`kis up` writes into server.py, so server.py owns them and passes them down as
arguments.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from kis.schema import Topology  # inlined by `kis`

WORK = Path("/kaggle/working")
PORT = 8080  # balancer; the only port the tunnel exposes
log = logging.getLogger("kis")
STARTED = time.time()


def log_file(name: str) -> IO[str]:
    """Log file for a child process; it stays open for the process lifetime."""
    return open(WORK / name, "w")


@dataclass
class Runtime:
    """The loaded preset and the layout of its backends."""

    preset: str = ""
    model: str = ""  # its alias
    topology: Topology = Topology.REPLICAS
    slots: int = 0  # parallel x instances
    parallel: int = 0  # slots per instance
    ctx: int = 0  # tokens per slot
