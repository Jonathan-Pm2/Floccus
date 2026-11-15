"""Project identity and the feedback link (no backend: feedback goes to GitHub issues)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from urllib.parse import urlencode

APP_NAME = "Floccus"
REPO_URL = "https://github.com/Jonathan-Pm2/Floccus"
GENTLE_SHELL_URL = "https://github.com/Gentleman-Programming/gentle-shell"
LICENSE = "Apache-2.0"
DOI = "10.5281/zenodo.21926426"  # concept DOI: always the latest version
DOI_URL = f"https://doi.org/{DOI}"
PRIMARY_COLOR = "#2e8b7a"
TEMPLATES = {"bug": "bug_report.yml", "idea": "feature_request.yml"}


def version() -> str:
    try:
        return _pkg_version("floccus-app")
    except PackageNotFoundError:
        return "dev"


def feedback_url(kind: str, page: str, lang: str) -> str:
    """Link to a prefilled GitHub issue form. Nothing is sent by the app itself."""
    query = urlencode({
        "template": TEMPLATES[kind],
        "version": version(),
        "page": page,
        "language": lang,
    })
    return f"{REPO_URL}/issues/new?{query}"
