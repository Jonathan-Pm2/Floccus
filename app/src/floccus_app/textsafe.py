"""Text that leaves the program: strip endpoints/credentials, escape Markdown."""

from __future__ import annotations

import re
from urllib.parse import quote

_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S+")
_HOST_ARG = re.compile(r"host='[^']*'")
_MD_SPECIAL = re.compile(r"([\\`*_\[\]()#!|>~<])")


def redact(text: str) -> str:
    """Remove URLs (with any user:token@host) and host='…' fragments from messages."""
    return _HOST_ARG.sub("host='<endpoint>'", _URL.sub("<endpoint>", str(text)))


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def doi_url(doi: str) -> str:
    return "https://doi.org/" + quote(str(doi).strip(), safe="/")
