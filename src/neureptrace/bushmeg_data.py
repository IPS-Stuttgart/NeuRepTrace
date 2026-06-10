"""Small authenticated BUSH-MEG data helper for NeuRepTrace smoke runs.

This module intentionally prepares only the files requested by the caller.  It
is meant for smoke tests and developer checks that should run without PyMEGDec;
it is not a full dataset synchronization client.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import HTTPBasicAuthHandler, HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, Request, build_opener
from xml.etree import ElementTree

from neureptrace.dataset_config import parse_participant_ids

BUSHMEG_WEBDAV_URL_ENV = "BUSHMEG_WEBDAV_URL"
BUSHMEG_DATA_KEY_ENV = "BUSHMEG_DATA_KEY"
BUSHMEG_DATA_PASSWORD_ENV = "BUSHMEG_DATA_PASSWORD"
BUSHMEG_DATA_DIR_ENV = "BUSHMEG_DATA_DIR"
BUSHMEG_REMOTE_PREFIX_ENV = "BUSHMEG_REMOTE_PREFIX"
BUSHMEG_DISCOVER_WEBDAV_ENV = "BUSHMEG_DISCOVER_WEBDAV"

DEFAULT_SMOKE_PARTICIPANTS = ("2",)
DEFAULT_SMOKE_ROLES = ("main", "cue")
FILE_TEMPLATES = {
    "main": "Part{participant}Data.mat",
    "cue": "Part{participant}CueData.mat",
}


@dataclass(frozen=True)
class BushMegFile:
    """One expected BUSH-MEG MATLAB file."""

    participant: str
    role: str
    relative_path: str
    local_path: Path
    exists: bool


def _participant_tokens(value: str | Iterable[str | int] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SMOKE_PARTICIPANTS
    parsed = parse_participant_ids(value)
    if not parsed:
        raise ValueError("At least one participant is required.")
    return tuple(str(participant) for participant in parsed)


def _role_tokens(value: Iterable[str] | None) -> tuple[str, ...]:
    roles = tuple(DEFAULT_SMOKE_ROLES if value is None else value)
    if not roles:
        raise ValueError("At least one file role is required.")
    unknown = sorted(set(roles) - set(FILE_TEMPLATES))
    if unknown:
        raise ValueError(f"Unknown BUSH-MEG file role(s): {', '.join(unknown)}. Available roles: {', '.join(sorted(FILE_TEMPLATES))}.")
    return roles


def expected_bushmeg_files(
    data_dir: str | Path,
    *,
    participants: str | Iterable[str | int] | None = None,
    roles: Iterable[str] | None = None,
    max_files: int | None = None,
) -> list[BushMegFile]:
    """Return the expected local files for a small BUSH-MEG run."""

    root = Path(data_dir)
    resolved: list[BushMegFile] = []
    for participant in _participant_tokens(participants):
        for role in _role_tokens(roles):
            relative_path = FILE_TEMPLATES[role].format(participant=participant)
            local_path = root / relative_path
            resolved.append(
                BushMegFile(
                    participant=participant,
                    role=role,
                    relative_path=relative_path,
                    local_path=local_path,
                    exists=local_path.exists(),
                )
            )
            if max_files is not None and len(resolved) >= int(max_files):
                return resolved
    return resolved


def _credentials_from_env(
    *,
    webdav_url: str | None,
    username: str | None,
    password: str | None,
) -> tuple[str | None, str | None, str | None]:
    return (
        webdav_url or os.environ.get(BUSHMEG_WEBDAV_URL_ENV),
        username or os.environ.get(BUSHMEG_DATA_KEY_ENV),
        password or os.environ.get(BUSHMEG_DATA_PASSWORD_ENV),
    )


def _require_credentials(webdav_url: str | None, username: str | None, password: str | None) -> tuple[str, str, str]:
    missing = []
    if not webdav_url:
        missing.append(BUSHMEG_WEBDAV_URL_ENV)
    if not username:
        missing.append(BUSHMEG_DATA_KEY_ENV)
    if not password:
        missing.append(BUSHMEG_DATA_PASSWORD_ENV)
    if missing:
        raise RuntimeError(
            "Missing BUSH-MEG WebDAV credential environment variable(s): "
            + ", ".join(missing)
            + ". Existing local files can still be used without credentials."
        )
    return str(webdav_url), str(username), str(password)


def _webdav_file_url(base_url: str, relative_path: str) -> str:
    base = base_url.rstrip("/") + "/"
    quoted = quote(relative_path.replace(os.sep, "/"), safe="/")
    return urljoin(base, quoted)


def _remote_prefixes(remote_prefix: str | None = None) -> tuple[str, ...]:
    configured = remote_prefix if remote_prefix is not None else os.environ.get(BUSHMEG_REMOTE_PREFIX_ENV, "")
    prefixes = [""]
    for raw in str(configured or "").replace(";", ",").split(","):
        prefix = raw.strip().strip("/")
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _candidate_remote_paths(relative_path: str, *, remote_prefix: str | None = None) -> tuple[str, ...]:
    paths = []
    for prefix in _remote_prefixes(remote_prefix):
        candidate = f"{prefix}/{relative_path}" if prefix else relative_path
        if candidate not in paths:
            paths.append(candidate)
    return tuple(paths)


def _build_webdav_opener(base_url: str, username: str, password: str):
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, base_url, username, password)
    return build_opener(
        HTTPBasicAuthHandler(password_manager),
        HTTPDigestAuthHandler(password_manager),
    )


class _PropfindRequest(Request):
    def get_method(self) -> str:
        return "PROPFIND"


def _discover_webdav_file(
    *,
    base_url: str,
    username: str,
    password: str,
    relative_path: str,
    timeout: float,
) -> str | None:
    """Return a discovered remote path for ``relative_path`` when WebDAV lists it.

    Some private WebDAV shares point the secret at a parent directory instead of
    the directory that directly contains the BUSH-MEG MATLAB files. A shallow
    PROPFIND gives CI a chance to discover the real path without logging the
    secret base URL.
    """

    opener = _build_webdav_opener(base_url, username, password)
    request = _PropfindRequest(base_url.rstrip("/") + "/", headers={"Depth": "1"})
    with opener.open(request, timeout=timeout) as response:
        payload = response.read()
    root = ElementTree.fromstring(payload)
    wanted_name = posixpath.basename(relative_path.replace(os.sep, "/"))
    base_path = urlparse(base_url.rstrip("/") + "/").path.rstrip("/") + "/"
    for element in root.iter():
        if not element.tag.endswith("href"):
            continue
        href = (element.text or "").strip()
        if not href:
            continue
        parsed = urlparse(href)
        href_path = unquote(parsed.path if parsed.scheme else href)
        if posixpath.basename(href_path.rstrip("/")) != wanted_name:
            continue
        if href_path.startswith(base_path):
            return href_path[len(base_path) :].lstrip("/")
        return href_path.lstrip("/")
    return None


def _download_webdav_file(
    *,
    base_url: str,
    username: str,
    password: str,
    relative_path: str,
    output_path: Path,
    timeout: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opener = _build_webdav_opener(base_url, username, password)
    url = _webdav_file_url(base_url, relative_path)

    with tempfile.NamedTemporaryFile(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with opener.open(url, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def prepare_bushmeg_smoke_data(
    data_dir: str | Path,
    *,
    participants: str | Iterable[str | int] | None = None,
    roles: Iterable[str] | None = None,
    max_files: int | None = None,
    webdav_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    remote_prefix: str | None = None,
    discover_webdav: bool | None = None,
    prefer_existing: bool = True,
    allow_missing: bool = False,
    timeout: float = 120.0,
) -> list[BushMegFile]:
    """Ensure that a small set of BUSH-MEG files exists locally.

    Existing files are always reused by default.  Missing files are downloaded
    from the configured WebDAV endpoint, but this helper never attempts to
    enumerate or synchronize the complete dataset.
    """

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    files = expected_bushmeg_files(
        data_root,
        participants=participants,
        roles=roles,
        max_files=max_files,
    )
    missing = [file for file in files if not file.exists]
    if not missing:
        return files

    resolved_url, resolved_username, resolved_password = _credentials_from_env(
        webdav_url=webdav_url,
        username=username,
        password=password,
    )
    try:
        resolved_url, resolved_username, resolved_password = _require_credentials(resolved_url, resolved_username, resolved_password)
    except RuntimeError:
        if allow_missing:
            return files
        raise

    for file in missing:
        if prefer_existing and file.local_path.exists():
            continue
        errors: list[str] = []
        remote_paths = list(_candidate_remote_paths(file.relative_path, remote_prefix=remote_prefix))
        should_discover = discover_webdav
        if should_discover is None:
            should_discover = os.environ.get(BUSHMEG_DISCOVER_WEBDAV_ENV, "true").lower() not in {"0", "false", "no", "off"}
        if should_discover:
            try:
                discovered = _discover_webdav_file(
                    base_url=resolved_url,
                    username=resolved_username,
                    password=resolved_password,
                    relative_path=file.relative_path,
                    timeout=timeout,
                )
            except Exception as exc:  # pragma: no cover - exercised only when private WebDAV permits PROPFIND.
                errors.append(f"PROPFIND discovery failed: {exc}")
            else:
                if discovered and discovered not in remote_paths:
                    remote_paths.append(discovered)

        for remote_path in remote_paths:
            try:
                _download_webdav_file(
                    base_url=resolved_url,
                    username=resolved_username,
                    password=resolved_password,
                    relative_path=remote_path,
                    output_path=file.local_path,
                    timeout=timeout,
                )
                break
            except HTTPError as exc:
                errors.append(f"{remote_path}: HTTP {exc.code} {exc.reason}")
            except Exception as exc:
                errors.append(f"{remote_path}: {exc}")
        if not file.local_path.exists() and not allow_missing:
            attempted = ", ".join(remote_paths)
            details = "; ".join(errors)
            raise RuntimeError(
                f"Could not download BUSH-MEG participant={file.participant} role={file.role} "
                f"to {file.local_path}. Tried remote path(s): {attempted}. {details}"
            )
    return expected_bushmeg_files(
        data_root,
        participants=participants,
        roles=roles,
        max_files=max_files,
    )


def _default_data_dir() -> Path:
    configured = os.environ.get(BUSHMEG_DATA_DIR_ENV)
    if configured:
        return Path(configured)
    return Path(".cache") / "bushmeg"


def _print_file_report(files: Sequence[BushMegFile]) -> None:
    for file in files:
        status = "present" if file.exists else "missing"
        print(f"{status}\tparticipant={file.participant}\trole={file.role}\t{file.local_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a limited BUSH-MEG data subset for NeuRepTrace smoke tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-smoke", help="Reuse or download the requested smoke-test files.")
    prepare.add_argument("--data-dir", type=Path, default=_default_data_dir())
    prepare.add_argument("--participants", default=",".join(DEFAULT_SMOKE_PARTICIPANTS), help="Participant ids/ranges, for example '2' or '1-2'.")
    prepare.add_argument("--roles", nargs="+", default=list(DEFAULT_SMOKE_ROLES), choices=sorted(FILE_TEMPLATES))
    prepare.add_argument("--max-files", type=int, default=None, help="Optional upper bound on downloaded/checked files.")
    prepare.add_argument("--webdav-url", default=None, help=f"Override {BUSHMEG_WEBDAV_URL_ENV}.")
    prepare.add_argument("--username", default=None, help=f"Override {BUSHMEG_DATA_KEY_ENV}.")
    prepare.add_argument("--password", default=None, help=f"Override {BUSHMEG_DATA_PASSWORD_ENV}. Prefer the environment variable in CI.")
    prepare.add_argument("--remote-prefix", default=None, help=f"Optional comma-separated remote subdirectory prefix(es), or {BUSHMEG_REMOTE_PREFIX_ENV}.")
    prepare.add_argument("--no-discover-webdav", action="store_true", help="Disable shallow WebDAV PROPFIND discovery for matching filenames.")
    prepare.add_argument("--allow-missing", action="store_true", help="Exit successfully even when requested private files cannot be downloaded.")
    prepare.add_argument("--timeout", type=float, default=120.0)

    list_files = subparsers.add_parser("list-smoke-files", help="List the local smoke-test files without downloading.")
    list_files.add_argument("--data-dir", type=Path, default=_default_data_dir())
    list_files.add_argument("--participants", default=",".join(DEFAULT_SMOKE_PARTICIPANTS))
    list_files.add_argument("--roles", nargs="+", default=list(DEFAULT_SMOKE_ROLES), choices=sorted(FILE_TEMPLATES))
    list_files.add_argument("--max-files", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "prepare-smoke":
        files = prepare_bushmeg_smoke_data(
            args.data_dir,
            participants=args.participants,
            roles=args.roles,
            max_files=args.max_files,
            webdav_url=args.webdav_url,
            username=args.username,
            password=args.password,
            remote_prefix=args.remote_prefix,
            discover_webdav=not args.no_discover_webdav,
            allow_missing=args.allow_missing,
            timeout=args.timeout,
        )
        _print_file_report(files)
        return 0 if args.allow_missing or all(file.exists for file in files) else 1

    if args.command == "list-smoke-files":
        files = expected_bushmeg_files(args.data_dir, participants=args.participants, roles=args.roles, max_files=args.max_files)
        _print_file_report(files)
        return 0 if all(file.exists for file in files) else 1

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
