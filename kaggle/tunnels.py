"""Tunnels that expose the balancer's port: cloudflared (quick or named) or tailscale.

Inlined into server.py by `kis` (see state.py). `notify` is passed in because it
needs the session's settings, which server.py owns.
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from kis.schema import EventType, Route, Tunnel, TunnelKind  # inlined by `kis`
from state import PORT, log, log_file  # inlined by `kis`

WATCH_SECONDS = 30  # between probes of the tunnel's public URL
WATCH_FAILURES = 3  # consecutive failed probes after which the tunnel is restarted


def download(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        os.chmod(path, 0o755)
    return path


def tunnel_alive(url: str, timeout: float = 15) -> bool:
    """True if `url` reaches this server through the tunnel: GET Route.HEALTH needs no key and does
    not count as activity. A 5xx from the edge (530 when no tunnel is registered), a DNS failure
    or a timeout means it does not."""
    try:
        urllib.request.urlopen(f"{url}{Route.HEALTH}", timeout=timeout).close()
    except urllib.error.HTTPError as e:
        return e.code < 500
    except OSError:
        return False
    return True


def watch_tunnel(url: str, proc: subprocess.Popen[str], interval: float = WATCH_SECONDS) -> None:
    """Kill cloudflared when its public URL stops reaching the balancer, so that the loop in
    cloudflared() restarts it and announces the new URL. A quick tunnel can lose its registration
    while the process keeps running, and clients would retry the dead URL until the session ends.
    Only a tunnel that worked once is watched: where the kernel cannot reach its own public URL,
    restarting would not help."""
    reached, failures = False, 0
    while proc.poll() is None:
        time.sleep(interval)
        if tunnel_alive(url):
            reached, failures = True, 0
        elif reached:
            failures += 1
            log.warning("tunnel %s unreachable (%d/%d)", url, failures, WATCH_FAILURES)
            if failures >= WATCH_FAILURES:
                log.error("tunnel %s is down: restarting cloudflared", url)
                proc.kill()
                return


def cloudflared(tunnel: Tunnel, on_ready: Callable[[str], None], notify: Callable[..., None]) -> None:
    """Quick tunnel (random trycloudflare.com URL) or, with a token, a named tunnel.

    A named tunnel's public hostname must point to http://localhost:8080.
    Runs forever, restarting the tunnel if it drops.
    """
    binary = download(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "/tmp/cloudflared"
    )
    cmd = [binary, "tunnel", "--no-autoupdate", "--protocol", "http2"]
    cmd += ["run", "--token", tunnel.token] if tunnel.token else ["--url", f"http://127.0.0.1:{PORT}"]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        announced = False
        for line in proc.stdout or []:
            log.info("cloudflared: %s", line.rstrip())
            quick = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            url = quick.group(0) if quick else (tunnel.public_url if "Registered tunnel connection" in line else None)
            if url and not announced:
                announced = True
                on_ready(url)
                threading.Thread(target=watch_tunnel, args=(url, proc), daemon=True).start()
        notify(EventType.TUNNEL_RESTART, code=proc.wait())
        time.sleep(5)


def tailscale(tunnel: Tunnel, on_ready: Callable[[str], None], notify: Callable[..., None]) -> None:
    """Join the tailnet in userspace mode; inbound tailnet connections reach localhost:<port>.
    Takes `notify` like every tunnel, but has no event of its own."""
    root = "/tmp/tailscale"
    if not os.path.exists(f"{root}/tailscale"):
        os.makedirs(root, exist_ok=True)
        subprocess.run(
            f"curl -fsSL https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
            f" | tar xz -C {root} --strip-components=1",
            shell=True,
            check=True,
        )
    ts = [f"{root}/tailscale", "--socket=/tmp/tailscaled.sock"]
    subprocess.Popen(
        [f"{root}/tailscaled", "--tun=userspace-networking", "--state=mem:", "--socket=/tmp/tailscaled.sock"],
        stdout=log_file("tailscaled.log"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    subprocess.run(
        [
            *ts,
            "up",
            f"--authkey={tunnel.authkey}",
            f"--hostname={tunnel.hostname}",
            "--accept-dns=false",
        ],
        check=True,
        timeout=120,
    )
    me = json.loads(subprocess.check_output([*ts, "status", "--json"]))["Self"]
    on_ready(f"http://{me['DNSName'].rstrip('.') or me['TailscaleIPs'][0]}:{PORT}")


TUNNELS = {TunnelKind.CLOUDFLARED: cloudflared, TunnelKind.TAILSCALE: tailscale}
