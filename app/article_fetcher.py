"""
Optional convenience: fetch a news article from a URL and extract its text.

Why this exists
---------------
Pasting a long article by hand is tedious, so the form offers an optional URL
field. Whatever text is fetched here is handed to the *existing* classifier
through ``detector.predict(title, body)``. Nothing about the machine-learning
pipeline changes: no retraining, and the reported metrics stay exactly as they
are. This module only retrieves text.

The security problem, and why it needs care
-------------------------------------------
Once the app fetches a URL supplied by a visitor, it becomes a server-side
request forgery (SSRF) risk. Without protection, a visitor could ask the server
to fetch things the public internet cannot reach:

    http://169.254.169.254/       cloud instance metadata (often holds credentials)
    http://127.0.0.1:5000/        the app's own admin endpoints
    http://10.0.0.5/              other machines on the private network
    file:///etc/passwd            local files

So every request is checked before a socket is opened:

1. The scheme must be http or https, which alone rules out ``file://``.
2. The hostname is resolved to IP addresses, and **every** address must be a
   public address. ``ipaddress.is_global`` is False for loopback, private,
   link-local and CGNAT ranges, which covers all of the targets above.
3. Redirects are followed *manually*, and each hop is re-validated. This step
   matters: an attacker could host a page on a public domain that redirects to
   169.254.169.254 and walk straight past a check that only looked at the
   original URL.

Nothing here writes to disk, executes the fetched text, or renders any part of
it as HTML.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# --- limits, chosen to keep a single-threaded deployment responsive ----------

TIMEOUT_SECONDS = 8          # Render's free tier runs one worker; long hangs block the site
MAX_BYTES = 2_000_000# refuse very large pages rather than filling memory
MAX_REDIRECTS = 3            # legitimate news sites rarely redirect more than once
# Below this many characters there is not enough text for a meaningful TF-IDF
# classification. Measured values: a BBC article gives about 2,700 characters,
# a news homepage gives about 230, and a 404 page gives about 130. Real articles
# are well above the threshold; error pages and index pages are well below it.
MIN_BODY_CHARACTERS = 400

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml", "application/xml", "text/plain")

# Elements that never contain the article body.
_CHROME_TAGS = ("script", "style", "nav", "footer", "aside", "form", "header", "noscript", "iframe")


# --------------------------------------------------------------------------
# Errors - one type per outcome, so the route can respond appropriately
# --------------------------------------------------------------------------

class FetchError(RuntimeError):
    """Base class for anything that stops an article being retrieved."""

    #: Shown to the user. Written to be read by a human, not a developer.
    user_message = "The article could not be retrieved from that link."

    #: Returned when this happens.
    status_code = 502


class InvalidURLError(FetchError):
    user_message = "That does not look like a valid web address. It should start with http:// or https://."
    status_code = 400


class BlockedAddressError(FetchError):
    user_message = (
        "That address points to a private or internal network, so it cannot be opened."
    )
    status_code = 400


class FetchTimeoutError(FetchError):
    user_message = (
        "The publisher took too long to respond. Paste the article text directly instead."
    )
    status_code = 504


class PublisherBlockedError(FetchError):
    user_message = (
        "This publisher does not allow automated reading, so the article could not be "
        "fetched. Please copy the article text and paste it into the box below."
    )
    status_code = 403


class PageNotFoundError(FetchError):
    user_message = "That page could not be found. Check the link and try again."
    status_code = 404


class NotHtmlError(FetchError):
    user_message = "That link did not point to a web page, so no article text was found."
    status_code = 415


class ExtractionError(FetchError):
    user_message = (
        "The page loaded but almost no article text could be read from it. It may need "
        "JavaScript, or the article may be behind a paywall. Please paste the text instead."
    )
    status_code = 422


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class FetchedArticle:
    """What we managed to retrieve from a URL."""

    title: str
    body: str
    final_url: str

    @property
    def character_count(self) -> int:
        return len(self.body)


# --------------------------------------------------------------------------
# SSRF protection
# --------------------------------------------------------------------------

def validate_public_url(url: str) -> str:
    """Return a normalised URL, or raise if it is unsafe to request.

    Raises
    ------
    InvalidURLError
        Not http/https, or no hostname.
    BlockedAddressError
        The hostname resolves to a loopback, private, link-local or CGNAT address.
    """
    if not url or not url.strip():
        raise InvalidURLError("No URL was provided.")

    candidate = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", candidate):
        # A bare "example.com/news" is a common paste, so assume https. But a
        # scheme-like prefix with no "//", such as "data:text/html,x" or
        # "javascript:alert(1)", is an unsupported scheme rather than a bare
        # host, and must be rejected rather than turned into an https URL.
        scheme_like = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):(?!//)", candidate)
        if scheme_like:
            raise InvalidURLError(
                f"Scheme '{scheme_like.group(1)}' is not allowed; use http or https."
            )
        candidate = "https://" + candidate

    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise InvalidURLError(str(exc)) from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise InvalidURLError(f"Scheme '{parsed.scheme}' is not allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise InvalidURLError("No hostname in that address.")

    # parsed.port raises ValueError for a malformed port such as "host:abc".
    # Without this guard a crafted URL would produce a confusing 500.
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise InvalidURLError(f"Invalid port in that address: {exc}") from exc
    port = parsed_port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise InvalidURLError(f"Could not resolve '{hostname}': {exc}") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise InvalidURLError(f"Could not resolve '{hostname}'.")

    # Every address must be public. A hostname resolving to both a public and a
    # private address is treated as hostile, because which one is used is not
    # something we control.
    for raw_ip in addresses:
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:  # pragma: no cover - getaddrinfo should give valid IPs
            raise InvalidURLError(f"Unrecognised address '{raw_ip}'.") from exc
        if not ip.is_global:
            raise BlockedAddressError(f"{hostname} resolves to {ip}, which is not public.")

    logger.info("URL approved for fetching: %s -> %s", hostname, sorted(addresses))
    return candidate


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_page(url: str, session: requests.Session | None = None) -> tuple[str, str]:
    """Download a page, following redirects safely.

    Returns ``(html, final_url)``.
    """
    session = session or requests.Session()
    current = url

    for _ in range(MAX_REDIRECTS + 1):
        # Re-validate on every hop. Skipping this lets a public URL redirect to
        # an internal address and bypass the first check entirely.
        current = validate_public_url(current)

        try:
            response = session.get(
                current,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=TIMEOUT_SECONDS,
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout as exc:
            raise FetchTimeoutError(str(exc)) from exc
        except requests.RequestException as exc:
            raise FetchError(str(exc)) from exc

        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise FetchError("Redirect with no target.")
                current = urljoin(current, location)
                logger.info("Following redirect to %s", current)
                continue

            if response.status_code in (401, 403, 429):
                raise PublisherBlockedError(f"HTTP {response.status_code}")
            if response.status_code == 404:
                raise PageNotFoundError("HTTP 404")
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code}")

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(ok in content_type for ok in ALLOWED_CONTENT_TYPES):
                raise NotHtmlError(content_type)

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    logger.warning("Truncating %s at %d bytes", current, MAX_BYTES)
                    break
                chunks.append(chunk)

            # requests already honours the HTTP charset; fall back to utf-8 and
            # ignore undecodable bytes rather than raising.
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"), current
        finally:
            response.close()

    raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS}).")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _strip_chrome(soup: BeautifulSoup) -> None:
    """Remove navigation, scripts and other elements that never hold the story."""
    for tag in _CHROME_TAGS:
        for element in soup.find_all(tag):
            element.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    """Headline from <h1>, falling back to <title> and then og:title."""
    for finder in (
        lambda: soup.find("h1"),
        lambda: soup.find("meta", attrs={"property": "og:title"}),
        lambda: soup.find("title"),
    ):
        try:
            element = finder()
        except Exception:  # pragma: no cover - defensive
            continue
        if element is None:
            continue
        if element.name == "meta":
            text = element.get("content", "")
        else:
            text = element.get_text(" ", strip=True)
        if text and len(text.strip()) > 3:
            return text.strip()
    return ""


def extract_article(html: str) -> tuple[str, str]:
    """Pull ``(title, body)`` out of an HTML document.

    The approach is a density heuristic: real article text sits inside one
    container element holding many substantial paragraphs, while menus and
    sidebars hold many short ones. So we score every paragraph's parent by the
    total text of all paragraphs underneath it and keep the winner.

    This is deliberately simple. It needs no extra dependency beyond
    BeautifulSoup and lxml, both already used elsewhere in the stack.
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_chrome(soup)

    title = _extract_title(soup)

    best_container = None
    best_score = 0
    for paragraph in soup.find_all("p"):
        container = paragraph.parent
        if container is None:
            continue
        score = sum(len(item.get_text(" ", strip=True)) for item in container.find_all("p"))
        if score > best_score:
            best_score = score
            best_container = container

    body = ""
    if best_container is not None:
        seen: set[int] = set()
        pieces: list[str] = []
        for paragraph in best_container.find_all("p"):
            # Deduplicate by identity: nested containers can yield the same <p> twice.
            if id(paragraph) in seen:
                continue
            seen.add(id(paragraph))
            text = paragraph.get_text(" ", strip=True)
            if len(text) > 1:
                pieces.append(text)
        body = " ".join(pieces)

    body = re.sub(r"\s{2,}", " ", body).strip()
    return title, body


def fetch_article(url: str, session: requests.Session | None = None) -> FetchedArticle:
    """Fetch a URL and extract its headline and body.

    Raises a :class:`FetchError` subclass with a message written for the user.
    """
    safe_url = validate_public_url(url)
    logger.info("Fetching article: %s", safe_url)

    html, final_url = fetch_page(safe_url, session=session)
    title, body = extract_article(html)

    # The important guard. A 404 or a paywall stub still contains enough
    # boilerplate for the model to happily return a confident verdict, which
    # would be meaningless. Refuse instead of guessing.
    if len(body) < MIN_BODY_CHARACTERS:
        raise ExtractionError(
            f"Only {len(body)} characters of article text were found "
            f"(minimum {MIN_BODY_CHARACTERS})."
        )

    result = FetchedArticle(title=title, body=body, final_url=final_url)
    logger.info(
        "Extracted %d characters (title: %r) from %s",
        result.character_count, title[:60], final_url,
    )
    return result