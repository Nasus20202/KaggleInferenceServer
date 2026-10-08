"""Client of the Kaggle server's admin routes (schema.Route)."""

import json
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from typing import Any

from . import retry
from .schema import Route, Stats, load

# urllib's default "Python-urllib/3.x" is blocked with a 403 (error 1010) by Cloudflare's
# Browser Integrity Check on a proxied hostname, as a named tunnel's is.
USER_AGENT = "kis (github.com/Nasus20202/KaggleInferenceServer)"


class AdminClient:
    """Calls one server endpoint with the API key; transient errors are retried."""

    def __init__(self, url: str, api_key: str, timeout: float = 30) -> None:
        self.url, self.api_key, self.timeout = url, api_key, timeout

    def _call(self, route: Route, method: str = "GET", **query: str | int) -> tuple[int, bytes]:
        """(HTTP status, body); raises OSError if the server can't be reached."""
        url = f"{self.url}{route}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        req = urllib.request.Request(
            url, method=method, headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT}
        )
        try:
            with retry.urlopen(req, timeout=self.timeout, what=route) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def stats(self) -> Stats | None:
        """Live stats, or None if the server answered with an error (OSError if unreachable)."""
        code, body = self._call(Route.STATS)
        return load(Stats, json.loads(body), "stats", strict=False) if code == HTTPStatus.OK else None

    def load(self, model: str) -> tuple[HTTPStatus, dict[str, Any]]:
        """Ask the server to swap to `model` (a preset name or alias): OK if it is loaded,
        ACCEPTED while it loads, NOT_FOUND if the session has no such preset."""
        code, body = self._call(Route.LOAD, "POST", model=model)
        try:
            answer = json.loads(body)
        except ValueError:
            answer = {"error": {"message": body.decode(errors="replace")}}
        return HTTPStatus(code), answer

    def models(self) -> list[dict[str, Any]]:
        """The session's presets as /v1/models lists them (id, preset, status); raises
        RuntimeError with the server's answer."""
        code, body = self._call(Route.MODELS)
        if code != HTTPStatus.OK:
            raise RuntimeError(body.decode(errors="replace"))
        return json.loads(body)["data"]

    def shutdown(self, session: str) -> HTTPStatus:
        """Ask `session` to stop: OK, or CONFLICT if the URL reached another session."""
        return HTTPStatus(self._call(Route.SHUTDOWN, "POST", session=session)[0])

    def logs(self, file: str, lines: int) -> str:
        """The last lines of a server-side log; raises RuntimeError with the server's answer."""
        code, body = self._call(Route.LOGS, file=file, lines=lines)
        if code != HTTPStatus.OK:
            raise RuntimeError(body.decode(errors="replace"))
        return body.decode(errors="replace")
