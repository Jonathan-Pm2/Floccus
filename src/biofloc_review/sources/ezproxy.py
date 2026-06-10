"""EZproxy URL rewriting and session-cookie utilities for institutional access.

The Universidad de Salamanca runs EZproxy at ``usal.idm.oclc.org`` with the
canonical URL rewrite pattern: replace the dots in the publisher hostname
with hyphens, then append ``.usal.idm.oclc.org``. For example,

    https://www.sciencedirect.com/science/article/pii/S001457932034567X
    ↓
    https://www-sciencedirect-com.usal.idm.oclc.org/science/article/pii/S001457932034567X

Once authenticated via the Shibboleth login flow, the session cookies issued
by ``usal.idm.oclc.org`` authorize subsequent HTTPS requests to any
proxy-rewritten URL. Authentication itself is interactive (web form +
institutional credentials) and is not automated by this module: the user
logs in once in their browser, exports the session cookies via a standard
extension (e.g. ``Get cookies.txt LOCALLY`` for Firefox/Chrome), and passes
the exported file to this module. Sessions typically last 24 hours.

This module intentionally avoids:
  - Automated login flows that capture institutional credentials.
  - Any technique to bypass authentication (out of scope for a PRISMA
    reproducibility-grade pipeline).
"""
from __future__ import annotations

import http.cookiejar
import logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

log = logging.getLogger(__name__)


DEFAULT_PROXY_HOST = "usal.idm.oclc.org"


def rewrite_for_ezproxy(url: str, proxy_host: str = DEFAULT_PROXY_HOST) -> str:
    """Rewrite a publisher URL through the institutional EZproxy.

    >>> rewrite_for_ezproxy("https://www.scopus.com/record/123")
    'https://www-scopus-com.usal.idm.oclc.org/record/123'

    Idempotent: URLs already pointing to the proxy are returned unchanged.
    """
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.netloc:
        return url
    if parsed.netloc.endswith("." + proxy_host) or parsed.netloc == proxy_host:
        return url
    new_netloc = parsed.netloc.replace(".", "-") + "." + proxy_host
    return urlunparse(parsed._replace(netloc=new_netloc))


def load_cookies(cookies_file: Path) -> http.cookiejar.MozillaCookieJar:
    """Load Netscape-format cookies exported from a browser extension.

    The expected format is the same one produced by ``Get cookies.txt LOCALLY``,
    ``cookies.txt``, EditThisCookie, and similar tools: a tab-separated file
    with one cookie per line and a leading ``# Netscape HTTP Cookie File``
    comment.
    """
    if not cookies_file.exists():
        raise FileNotFoundError(f"Cookies file not found: {cookies_file}")
    jar = http.cookiejar.MozillaCookieJar()
    jar.load(str(cookies_file), ignore_discard=True, ignore_expires=True)
    log.info("Loaded %d cookies from %s", len(jar), cookies_file)
    return jar


def build_session(cookies_file: Path | None = None,
                   user_agent: str | None = None) -> requests.Session:
    """Build a ``requests.Session`` preloaded with proxy cookies if given.

    A browser-like User-Agent is set by default since several publishers
    serve trimmed-down responses or block requests with bot UAs even when
    coming through EZproxy.
    """
    session = requests.Session()
    if cookies_file is not None:
        jar = load_cookies(cookies_file)
        session.cookies = jar
    session.headers.update({
        "User-Agent": user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
            "Gecko/20100101 Firefox/120.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/pdf;q=0.9,"
            "application/xml;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.8,es;q=0.5,pt;q=0.3",
    })
    return session


def is_session_valid(session: requests.Session,
                      test_url: str = "https://www-sciencedirect-com.usal.idm.oclc.org/",
                      timeout: float = 15.0) -> bool:
    """Cheap liveness check: hit a proxy-rewritten URL and verify the
    response is not a Shibboleth login page (would indicate expired session).
    """
    try:
        r = session.get(test_url, timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        log.warning("Session liveness check failed: %s", e)
        return False
    if r.status_code != 200:
        return False
    # Shibboleth login pages return text/html with characteristic markers.
    text_lower = (r.text or "").lower()[:4000]
    looks_like_login = any(marker in text_lower for marker in (
        "shibboleth",
        "single sign-on",
        "sign in to your institution",
        "iniciar sesi&oacute;n",
        "iniciar sesión",
    ))
    if looks_like_login:
        log.info("Session appears to require re-authentication.")
        return False
    return True
