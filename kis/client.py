"""Client of the Kaggle server's admin routes (schema.Route)."""

import json
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus

from . import retry
from .schema import Route, Stats, load


class AdminClient:
    """Calls one server endpoint with the API key; transient errors are retried."""

    def __init__(self, url: str, api_key: str, timeout: float = 30) -> None:
        self.url, self.api_key, self.timeout = url, api_key, timeout

    def _call(self, route: Route, method: str = "GET", **query: str | int) -> tuple[int, bytes]:
        """(HTTP status, body); raises OSError if the server can't be reached."""
        url = f"{self.url}{route}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with retry.urlopen(req, timeout=self.timeout, what=route) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def stats(self) -> Stats | None:
        """Live stats, or None if the server answered with an error (OSError if unreachable)."""
        code, body = self._call(Route.STATS)
        return load(Stats, json.loads(body), "stats", strict=False) if code == HTTPStatus.OK else None

    def shutdown(self, session: str) -> HTTPStatus:
        """Ask `session` to stop: OK, or CONFLICT if the URL reached another session."""
        return HTTPStatus(self._call(Route.SHUTDOWN, "POST", session=session)[0])

    def logs(self, file: str, lines: int) -> str:
        """The last lines of a server-side log; raises RuntimeError with the server's answer."""
        code, body = self._call(Route.LOGS, file=file, lines=lines)
        if code != HTTPStatus.OK:
            raise RuntimeError(body.decode(errors="replace"))
        return body.decode(errors="replace")
