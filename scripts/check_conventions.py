"""Command-line checker for the writing conventions of the repository.

It verifies that comments and docstrings are self-contained, that every file starts
with an English header, that test files stay short and that doctests run. Each rule is
a function returning ``Violation`` objects; ``main`` is the entry point.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RULES: tuple[str, ...] = ("refs", "history", "header", "length", "doctest")
VERIFY_RULE = "verify"

# Périmètre par défaut des fichiers Python
DEFAULT_PYTHON_DIRS: tuple[str, ...] = ("tsforecast", "tests", "scripts")
CONFIG_SUFFIXES: frozenset[str] = frozenset({".yml", ".yaml", ".toml"})

# Dossiers ignorés : à n'importe quelle profondeur, ou seulement à la racine du dépôt
EXCLUDED_ANYWHERE: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}
)
EXCLUDED_AT_ROOT: frozenset[str] = frozenset({"site", "htmlcov", ".venv", "venv"})
EXCLUDED_PREFIXES: tuple[str, ...] = ("tests/support/golden",)

MAX_TEST_LINES = 1200
MAX_PACKAGE_LINES = 1500
MIN_HEADER_CHARS = 40
MIN_XFAIL_REASON_WORDS = 6

# Références à des documents ou numéros extérieurs au fichier
REF_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"§\s*\d",
        r"\b[Ss]ection\s+\d+(\.\d+)*",
        r"\bD\d{1,3}\b",
        r"\bI\d{1,2}\b",
        r"\bANO-[A-Z]+-\d+",
        r"ANOMALIES\.md",
        r"\[(CODE|DOC|SPEC)\]",
        r"\b(décision|decision|invariant|Invariant|Décision)\s+[A-Z]?\d+",
        r"\bprompt\s+[A-Z]\d+\b",
        r"high_frequency_imputer\d?_(architecture|prompts|review|annotations)",
        r"tests_and_refactoring_prompts",
        r"PLAN_B",
    )
) + (re.compile(r"\b(spec|spécification)\b.*(\d|§)", re.IGNORECASE),)

# Mots d'historique ou de justification de décision (relecture manuelle)
HISTORY_PATTERN = re.compile(
    r"\bauparavant\b|\banciennement\b|\bancien(?:ne)?s?\b|\bdésormais\b"
    r"|\bn['’]est plus\b|\bplus besoin\b|\bprécédemment\b|\bétait\b"
    r"|\bcorrig(?:é|ée|és|ées)\b|\bcorrection de\b|\bsuite à\b"
    r"|\bremplac(?:e|é|ée|és|ées)\b|\bcontrairement à\b|\bworkaround\b"
    r"|\bpreviously\b|\bformerly\b|\bused to\b|\bno longer\b|\blegacy\b"
    r"|\bfixed\b|\binstead of the old\b|\bwas changed\b|\bnow returns\b",
    re.IGNORECASE,
)

# Heuristique d'anglais : accents et mots français courants
FRENCH_ACCENT_PATTERN = re.compile(r"[àâçéèêëîïôùûü]", re.IGNORECASE)
FRENCH_WORD_PATTERN = re.compile(r"\b(le|la|les|des|pour|dans|avec|une?)\b", re.IGNORECASE)

# Commentaires de directive d'outil, exemptés de la règle refs (ex. codes de lint)
DIRECTIVE_PATTERN = re.compile(
    r"^\s*(noqa|type:|pragma|pylint:|fmt:|isort:|ruff:|mypy:)", re.IGNORECASE
)
ANO_PATTERN = re.compile(r"ANO-[\w-]+")


@dataclass(frozen=True)
class Violation:
    """One finding of a rule.

    Attributes:
        rule: Rule name (``refs``, ``history``, ``header``, ``length``, ``doctest``
            or ``verify``).
        path: File path, relative to the repository root, with forward slashes.
        line: 1-based line number (0 when not applicable).
        message: Human-readable description.
        blocking: Whether the finding makes the exit code non-zero.
        context: Offending source line, when relevant.
    """

    rule: str
    path: str
    line: int
    message: str
    blocking: bool = True
    context: str = ""


# ---------------------------------------------------------------------------
# Collecte des fichiers
# ---------------------------------------------------------------------------


def relative_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` with forward slashes.

    Args:
        path: File or directory path.
        root: Repository root.

    Returns:
        The relative POSIX path, or the path itself when outside ``root``.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_excluded(relative: str) -> bool:
    """Tell whether a repository-relative path is outside the checked scope.

    Args:
        relative: POSIX path relative to the repository root.

    Returns:
        True for cache folders, virtual environments, generated sites and golden files.
    """
    parts = relative.split("/")
    if parts and parts[0] in EXCLUDED_AT_ROOT:
        return True
    if any(part in EXCLUDED_ANYWHERE for part in parts):
        return True
    return any(
        relative == prefix or relative.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES
    )


def collect_files(root: Path, targets: Sequence[Path], suffixes: frozenset[str]) -> list[Path]:
    """List the files of the given targets that have one of the suffixes.

    Args:
        root: Repository root (for the exclusions).
        targets: Absolute files or directories; missing ones are skipped.
        suffixes: Accepted file suffixes, with the leading dot.

    Returns:
        Sorted list of absolute file paths.
    """
    found: set[Path] = set()
    for target in targets:
        if target.is_file():
            if target.suffix in suffixes and not is_excluded(relative_path(target, root)):
                found.add(target.resolve())
            continue
        for current, dirnames, filenames in os.walk(target):
            current_path = Path(current)
            # Élagage des dossiers exclus pour ne pas les parcourir
            dirnames[:] = [
                name
                for name in dirnames
                if not is_excluded(relative_path(current_path / name, root))
            ]
            for name in filenames:
                candidate = current_path / name
                if candidate.suffix in suffixes and not is_excluded(
                    relative_path(candidate, root)
                ):
                    found.add(candidate.resolve())
    return sorted(found)


def directory_key(relative: str) -> str:
    """Group key of a file for the summary table.

    Args:
        relative: POSIX path relative to the repository root.

    Returns:
        Two path components of the parent directory (three under ``tests``),
        or ``.`` for a file at the root.
    """
    parts = relative.split("/")[:-1]
    depth = 3 if parts and parts[0] == "tests" else 2
    return "/".join(parts[:depth]) or "."


# ---------------------------------------------------------------------------
# Extraction des commentaires et docstrings
# ---------------------------------------------------------------------------


def _docstring_node(node: ast.AST) -> Optional[ast.Constant]:
    """Return the docstring constant of a module, class or function node, if any."""
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value
    return None


def iter_text_lines(data: bytes) -> list[tuple[int, str]]:
    """Extract the comment lines and docstring lines of Python source.

    Only comments (via ``tokenize``) and docstrings of modules, classes and functions
    (via ``ast``) are returned; other string literals are ignored.

    Args:
        data: Python source as bytes.

    Returns:
        ``(line number, text)`` pairs, one per comment and per docstring line, sorted
        by line number. Tool-directive comments (``noqa``, ``type:`` ...) are skipped.

    Raises:
        SyntaxError: If the source cannot be parsed.
        tokenize.TokenError: If the source cannot be tokenized.
    """
    tree = ast.parse(data)
    lines: list[tuple[int, str]] = []
    for token in tokenize.tokenize(io.BytesIO(data).readline):
        if token.type == tokenize.COMMENT:
            text = token.string.lstrip("#")
            if not DIRECTIVE_PATTERN.match(text):
                lines.append((token.start[0], text))
    for node in ast.walk(tree):
        constant = _docstring_node(node)
        if constant is not None:
            for offset, text in enumerate(str(constant.value).splitlines()):
                lines.append((constant.lineno + offset, text))
    return sorted(lines)


def _read_source(path: Path) -> bytes:
    """Read a source file as bytes."""
    return path.read_bytes()


def _syntax_violation(rule: str, relative: str, error: Exception) -> Violation:
    """Build the violation reported for an unparsable file."""
    line = getattr(error, "lineno", None) or 0
    return Violation(rule, relative, line, f"fichier illisible : {error}")


# ---------------------------------------------------------------------------
# Règle refs
# ---------------------------------------------------------------------------


def _xfail_violations(tree: ast.AST, relative: str) -> list[Violation]:
    """Check the ``reason=`` of xfail markers that cite an anomaly identifier."""
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "xfail":
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg != "reason" or not isinstance(value, ast.Constant):
                continue
            if not isinstance(value.value, str) or not ANO_PATTERN.search(value.value):
                continue
            remainder = ANO_PATTERN.sub(" ", value.value)
            if len(re.findall(r"\w+", remainder)) < MIN_XFAIL_REASON_WORDS:
                violations.append(
                    Violation(
                        "refs",
                        relative,
                        value.lineno,
                        f"reason= d'xfail trop court (au moins {MIN_XFAIL_REASON_WORDS} mots "
                        "après l'identifiant)",
                        context=value.value,
                    )
                )
    return violations


def check_refs_file(path: Path, root: Path) -> list[Violation]:
    """Look for external references in the comments and docstrings of one file.

    Args:
        path: Python file to check.
        root: Repository root, for the displayed path.

    Returns:
        One violation per forbidden pattern and per line, plus xfail ``reason=`` issues.
    """
    relative = relative_path(path, root)
    data = _read_source(path)
    try:
        text_lines = iter_text_lines(data)
        tree = ast.parse(data)
    except (SyntaxError, tokenize.TokenError) as error:
        return [_syntax_violation("refs", relative, error)]
    violations = []
    for lineno, text in text_lines:
        for pattern in REF_PATTERNS:
            match = pattern.search(text)
            if match:
                excerpt = match.group(0).strip()[:40]
                violations.append(
                    Violation(
                        "refs",
                        relative,
                        lineno,
                        f"référence extérieure interdite : « {excerpt} »",
                        context=text.strip(),
                    )
                )
    violations.extend(_xfail_violations(tree, relative))
    return violations


# ---------------------------------------------------------------------------
# Règle history
# ---------------------------------------------------------------------------


def check_history_file(path: Path, root: Path) -> list[Violation]:
    """Flag history or decision-justification wording in one file (non blocking).

    Args:
        path: Python file to check.
        root: Repository root, for the displayed path.

    Returns:
        One non-blocking violation per matching keyword and per line.
    """
    relative = relative_path(path, root)
    try:
        text_lines = iter_text_lines(_read_source(path))
    except (SyntaxError, tokenize.TokenError) as error:
        return [_syntax_violation("history", relative, error)]
    violations = []
    for lineno, text in text_lines:
        keywords = sorted({m.group(0).lower() for m in HISTORY_PATTERN.finditer(text)})
        for keyword in keywords:
            violations.append(
                Violation(
                    "history",
                    relative,
                    lineno,
                    f"à relire : mot d'historique « {keyword} »",
                    blocking=False,
                    context=text.strip(),
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Règle header
# ---------------------------------------------------------------------------


def _english_problem(text: str) -> Optional[str]:
    """Return the reason why a text does not look English, or None."""
    accent = FRENCH_ACCENT_PATTERN.search(text)
    if accent:
        return f"caractère accentué « {accent.group(0)} »"
    word = FRENCH_WORD_PATTERN.search(text)
    if word:
        return f"mot français « {word.group(0)} »"
    return None


def check_python_header(path: Path, root: Path) -> list[Violation]:
    """Check the module docstring of one Python file.

    The first statement must be a docstring of at least ``MIN_HEADER_CHARS``
    characters, written in English and free of external references.

    Args:
        path: Python file to check.
        root: Repository root, for the displayed path.

    Returns:
        Violations found (empty when the header is valid).
    """
    relative = relative_path(path, root)
    try:
        tree = ast.parse(_read_source(path))
    except SyntaxError as error:
        return [_syntax_violation("header", relative, error)]
    constant = _docstring_node(tree)
    if constant is None:
        return [Violation("header", relative, 1, "docstring de module absente en première instruction")]
    text = str(constant.value).strip()
    if len(text) < MIN_HEADER_CHARS:
        return [
            Violation(
                "header",
                relative,
                constant.lineno,
                f"en-tête trop court ({len(text)} < {MIN_HEADER_CHARS} caractères)",
            )
        ]
    problem = _english_problem(text)
    if problem:
        return [
            Violation(
                "header", relative, constant.lineno, f"en-tête non anglais : {problem}"
            )
        ]
    for pattern in REF_PATTERNS:
        match = pattern.search(text)
        if match:
            return [
                Violation(
                    "header",
                    relative,
                    constant.lineno,
                    f"en-tête avec référence extérieure : « {match.group(0).strip()[:40]} »",
                )
            ]
    return []


def check_config_header(path: Path, root: Path) -> list[Violation]:
    """Check that a YAML or TOML file starts with a ``#`` comment.

    A shebang line and ``# yaml-language-server`` lines are tolerated above it.

    Args:
        path: Configuration file to check.
        root: Repository root, for the displayed path.

    Returns:
        Violations found (empty when the first meaningful line is a comment).
    """
    relative = relative_path(path, root)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#!") or stripped.startswith("# yaml-language-server"):
            continue
        if stripped.startswith("#"):
            return []
        return [
            Violation("header", relative, lineno, "commentaire d'en-tête « # » absent", context=stripped)
        ]
    return [Violation("header", relative, 1, "fichier vide : commentaire d'en-tête absent")]


# ---------------------------------------------------------------------------
# Règle length
# ---------------------------------------------------------------------------


def check_length_file(path: Path, root: Path) -> list[Violation]:
    """Check the length of one Python file.

    Test files above ``MAX_TEST_LINES`` lines are blocking; package files above
    ``MAX_PACKAGE_LINES`` lines are informational only.

    Args:
        path: Python file to check.
        root: Repository root, for the displayed path.

    Returns:
        Violations found (empty when the file is short enough).
    """
    relative = relative_path(path, root)
    count = len(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())
    if relative.startswith("tests/") and count > MAX_TEST_LINES:
        return [
            Violation(
                "length", relative, 0, f"fichier de tests trop long ({count} > {MAX_TEST_LINES} lignes)"
            )
        ]
    if relative.startswith("tsforecast/") and count > MAX_PACKAGE_LINES:
        return [
            Violation(
                "length",
                relative,
                0,
                f"informatif : fichier long ({count} > {MAX_PACKAGE_LINES} lignes)",
                blocking=False,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Règle doctest
# ---------------------------------------------------------------------------


def run_doctests(root: Path, targets: Sequence[str]) -> list[Violation]:
    """Run ``pytest --doctest-modules`` and report the failing doctests.

    Args:
        root: Directory in which pytest is launched.
        targets: Files or directories passed to pytest.

    Returns:
        One violation per failing doctest or collection error, or a single
        violation carrying the output tail when pytest fails without a parsable report.
    """
    command = [sys.executable, "-m", "pytest", "--doctest-modules", *targets, "-q", "-rfE"]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    violations = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(FAILED|ERROR) (\S+?)(?:::(\S+))?(?: - (.*))?$", line)
        if match:
            path = match.group(2).replace("\\", "/")
            detail = match.group(4) or match.group(3) or ""
            violations.append(Violation("doctest", path, 0, f"{match.group(1)} {detail}".strip()))
    # Code 5 : aucun doctest collecté, ce qui n'est pas une erreur
    if result.returncode not in (0, 5) and not violations:
        tail = "\n".join(result.stdout.splitlines()[-10:] + result.stderr.splitlines()[-5:])
        violations.append(Violation("doctest", ".", 0, "pytest a échoué", context=tail))
    return violations


# ---------------------------------------------------------------------------
# Option --verify-code-unchanged
# ---------------------------------------------------------------------------


class _DocstringStripper(ast.NodeTransformer):
    """AST transformer removing the docstrings of modules, classes and functions."""

    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        if _docstring_node(node) is not None:
            body = node.body[1:]  # type: ignore[attr-defined]
            # Un corps de classe ou de fonction ne peut pas être vide ; un module le peut
            if not body and not isinstance(node, ast.Module):
                body = [ast.Pass()]
            node.body = body  # type: ignore[attr-defined]
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def dump_without_docstrings(source: bytes) -> str:
    """Dump the AST of Python source after removing all docstrings.

    Args:
        source: Python source as bytes.

    Returns:
        Indented ``ast.dump`` text, suitable for diffing.

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
    tree = _DocstringStripper().visit(ast.parse(source))
    return ast.dump(tree, indent=1)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run a git command in ``root`` and return the completed process."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True)


def modified_python_files(root: Path, ref: str) -> list[str]:
    """List the Python files modified (not created, not deleted) since a git ref.

    Args:
        root: Repository root.
        ref: Git reference to compare against.

    Returns:
        POSIX paths relative to ``root``.

    Raises:
        RuntimeError: If git fails.
    """
    result = _git(root, "diff", "--name-status", "-z", "--diff-filter=M", ref, "--")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    tokens = result.stdout.decode("utf-8", errors="replace").split("\0")
    paths = []
    # Format -z : statut puis chemin, séparés par des octets nuls
    for status, name in zip(tokens[0::2], tokens[1::2]):
        if status.startswith("M") and name.endswith(".py"):
            paths.append(name)
    return paths


def check_code_unchanged(relative: str, root: Path, ref: str) -> list[Violation]:
    """Compare the code of a file with its version at a git ref, ignoring docstrings.

    Args:
        relative: POSIX path of the file, relative to ``root``.
        root: Repository root.
        ref: Git reference holding the previous version.

    Returns:
        A violation carrying the AST diff when the code differs, otherwise empty.
    """
    old = _git(root, "show", f"{ref}:{relative}")
    if old.returncode != 0:
        return []
    try:
        before = dump_without_docstrings(old.stdout)
        after = dump_without_docstrings((root / relative).read_bytes())
    except SyntaxError as error:
        return [_syntax_violation(VERIFY_RULE, relative, error)]
    if before == after:
        return []
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), f"{ref}:{relative}", relative, lineterm="", n=2
        )
    )
    return [Violation(VERIFY_RULE, relative, 0, "le code a changé hors docstrings", context=diff)]


def verify_code_unchanged(root: Path, ref: str, scope: Sequence[Path]) -> list[Violation]:
    """Check that the modified files in scope differ from ``ref`` only by docstrings.

    Args:
        root: Repository root.
        ref: Git reference to compare against.
        scope: Absolute Python files in the checked scope.

    Returns:
        All code-change violations; a single violation when git itself fails.
    """
    try:
        modified = modified_python_files(root, ref)
    except RuntimeError as error:
        return [Violation(VERIFY_RULE, ".", 0, f"git a échoué : {error}")]
    in_scope = {relative_path(path, root) for path in scope}
    violations: list[Violation] = []
    for relative in modified:
        if relative in in_scope:
            violations.extend(check_code_unchanged(relative, root, ref))
    return violations


# ---------------------------------------------------------------------------
# Orchestration et sortie
# ---------------------------------------------------------------------------


def run_rules(
    root: Path, rules: Sequence[str], paths: Sequence[Path], verify_ref: Optional[str]
) -> list[Violation]:
    """Run the selected rules over the scope.

    Args:
        root: Repository root.
        rules: Rule names to run (subset of ``RULES``).
        paths: Absolute targets; empty to use the default scope.
        verify_ref: Git ref for the code-unchanged check, or None to skip it.

    Returns:
        All violations, blocking or not.
    """
    python_targets = list(paths) or [root / name for name in DEFAULT_PYTHON_DIRS]
    config_targets = list(paths) or [root]
    python_files = collect_files(root, python_targets, frozenset({".py"}))
    violations: list[Violation] = []
    per_file = {
        "refs": check_refs_file,
        "history": check_history_file,
        "length": check_length_file,
    }
    for rule in rules:
        if rule in per_file:
            for path in python_files:
                violations.extend(per_file[rule](path, root))
        elif rule == "header":
            for path in python_files:
                violations.extend(check_python_header(path, root))
            for path in collect_files(root, config_targets, CONFIG_SUFFIXES):
                violations.extend(check_config_header(path, root))
        elif rule == "doctest":
            targets = [relative_path(path, root) for path in paths] or ["tsforecast"]
            violations.extend(run_doctests(root, targets))
    if verify_ref is not None:
        violations.extend(verify_code_unchanged(root, verify_ref, python_files))
    return violations


def format_summary(violations: Sequence[Violation], rules: Sequence[str]) -> str:
    """Build the table of violation counts per directory and per rule.

    Args:
        violations: Violations to count.
        rules: Rule names to show as columns, in order.

    Returns:
        The table as text, with a final total row.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for violation in violations:
        counts[directory_key(violation.path)][violation.rule] += 1
    labels = {rule: ("history (à relire)" if rule == "history" else rule) for rule in rules}
    headers = ["dossier", *(labels[rule] for rule in rules), "total"]
    rows = []
    totals = {rule: 0 for rule in rules}
    for directory in sorted(counts):
        cells = [counts[directory][rule] for rule in rules]
        for rule, value in zip(rules, cells):
            totals[rule] += value
        rows.append([directory, *map(str, cells), str(sum(cells))])
    rows.append(["TOTAL", *(str(totals[rule]) for rule in rules), str(sum(totals.values()))])
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(
                cell.ljust(width) if i == 0 else cell.rjust(width)
                for i, (cell, width) in enumerate(zip(row, widths))
            )
        )
    return "\n".join(lines)


def format_detail(violations: Sequence[Violation]) -> str:
    """List the violations one per line, with their context.

    Args:
        violations: Violations to display.

    Returns:
        The listing as text (empty string when there is none).
    """
    lines = []
    for violation in sorted(violations, key=lambda v: (v.rule, v.path, v.line)):
        marker = "" if violation.blocking else " (non bloquant)"
        lines.append(
            f"{violation.path}:{violation.line}: [{violation.rule}] {violation.message}{marker}"
        )
        if violation.context:
            lines.extend(f"    {context_line}" for context_line in violation.context.splitlines())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Vérifie les conventions de rédaction du dépôt")
    parser.add_argument("--rule", choices=[*RULES, "all"], default=None, help="Règle à exécuter")
    parser.add_argument("--format", choices=["summary", "detail"], default="summary")
    parser.add_argument(
        "--verify-code-unchanged",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="REF",
        help="Vérifie que seuls les docstrings diffèrent du ref git (défaut HEAD)",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("paths", nargs="*", type=Path, help="Fichiers ou dossiers à vérifier")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the checker.

    Without ``--rule``, all rules run, except when ``--verify-code-unchanged`` is
    given alone: only the code comparison then runs.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when None).

    Returns:
        1 if a blocking rule is violated, 0 otherwise.
    """
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = args.root.resolve()
    paths = [path.resolve() for path in args.paths]
    if args.rule is None:
        rules = [] if args.verify_code_unchanged is not None else list(RULES)
    elif args.rule == "all":
        rules = list(RULES)
    else:
        rules = [args.rule]
    violations = run_rules(root, rules, paths, args.verify_code_unchanged)
    shown_rules = [*rules, *([VERIFY_RULE] if args.verify_code_unchanged is not None else [])]
    if args.format == "detail":
        detail = format_detail(violations)
        if detail:
            print(detail)
            print()
    else:
        informative = [v for v in violations if not v.blocking and v.rule == "length"]
        for violation in informative:
            print(f"{violation.path}: {violation.message}")
    print(format_summary(violations, shown_rules))
    blocking = sum(1 for violation in violations if violation.blocking)
    print(f"\nViolations bloquantes : {blocking}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
