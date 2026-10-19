"""Tests of ``scripts/check_conventions.py``: one family of scenarios per rule (passing case, failing case, avoided false positive), file collection, summary output and the code-unchanged check on a temporary git repository."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import check_conventions as cc

GOOD_HEADER = '"""Utilities for the tests of the checker, with a header long enough."""\n'


def write(root: Path, relative: str, text: str) -> Path:
    """Write a file under ``root`` and return its path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def refs(tmp_path: Path, body: str) -> list[cc.Violation]:
    """Run the refs rule on a file made of the given body."""
    return cc.check_refs_file(write(tmp_path, "mod.py", body), tmp_path)


# ---------------------------------------------------------------------------
# refs
# ---------------------------------------------------------------------------


class TestRefs:
    def test_clean_file_passes(self, tmp_path):
        assert refs(tmp_path, GOOD_HEADER + "# Vérification des arguments\nx = 1\n") == []

    @pytest.mark.parametrize(
        "comment",
        [
            "# voir § 3.2",
            "# Section 4 du document",
            "# décision D47",
            "# invariant I11",
            "# anomalie ANO-FREQ-073",
            "# voir ANOMALIES.md",
            "# [CODE] problème connu",
            "# décision 12",
            "# prompt N3",
            "# voir high_frequency_imputer2_architecture",
            "# voir tests_and_refactoring_prompts",
            "# voir PLAN_B",
            "# voir la spec section 3",
            "# la spécification § 2",
        ],
    )
    def test_comment_reference_is_flagged(self, tmp_path, comment):
        violations = refs(tmp_path, f"{GOOD_HEADER}{comment}\nx = 1\n")
        assert len(violations) >= 1
        assert all(v.rule == "refs" and v.blocking for v in violations)
        assert violations[0].line == 2

    def test_docstring_reference_is_flagged_with_its_line(self, tmp_path):
        body = 'def f():\n    """Summary.\n\n    Follows D47 here.\n    """\n'
        violations = refs(tmp_path, body)
        assert [v.line for v in violations] == [4]

    def test_class_docstring_is_scanned(self, tmp_path):
        violations = refs(tmp_path, 'class A:\n    """See § 3."""\n')
        assert len(violations) == 1

    def test_plain_string_is_not_scanned(self, tmp_path):
        body = 'label = "D1"\nname = "ANO-X-1 § 3"\ntext = """\nD47 and §3\n"""\n'
        assert refs(tmp_path, body) == []

    def test_hash_inside_string_is_not_a_comment(self, tmp_path):
        assert refs(tmp_path, 'x = "# D47"\n') == []

    def test_spec_without_digit_is_accepted(self, tmp_path):
        assert refs(tmp_path, "# conforme à la spec du format\nx = 1\n") == []

    def test_lint_directive_is_ignored(self, tmp_path):
        assert refs(tmp_path, "x = 1  # noqa: D100\n") == []

    def test_unparsable_file_is_reported(self, tmp_path):
        violations = refs(tmp_path, "def broken(:\n")
        assert len(violations) == 1 and "illisible" in violations[0].message

    def test_xfail_reason_compliant(self, tmp_path):
        body = (
            "import pytest\n"
            '@pytest.mark.xfail(reason="ANO-FREQ-001 la fréquence est mal détectée sur un index court")\n'
            "def test_a(): ...\n"
        )
        assert refs(tmp_path, body) == []

    def test_xfail_reason_too_short(self, tmp_path):
        body = (
            "import pytest\n"
            '@pytest.mark.xfail(reason="ANO-FREQ-001 bug connu")\n'
            "def test_a(): ...\n"
        )
        violations = refs(tmp_path, body)
        assert len(violations) == 1 and "xfail" in violations[0].message

    def test_xfail_reason_without_identifier_is_not_checked(self, tmp_path):
        body = 'import pytest\n@pytest.mark.xfail(reason="court")\ndef test_a(): ...\n'
        assert refs(tmp_path, body) == []


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


class TestHistory:
    def history(self, tmp_path, body):
        return cc.check_history_file(write(tmp_path, "mod.py", body), tmp_path)

    def test_clean_file_passes(self, tmp_path):
        assert self.history(tmp_path, GOOD_HEADER + "# Calcul exact en entiers\n") == []

    @pytest.mark.parametrize(
        "comment, keyword",
        [
            ("# auparavant on triait", "auparavant"),
            ("# désormais trié", "désormais"),
            ("# l'ancienne version", "ancienne"),
            ("# previously sorted", "previously"),
            ("# no longer needed", "no longer"),
            ("# workaround pour pandas", "workaround"),
        ],
    )
    def test_keyword_is_flagged_non_blocking(self, tmp_path, comment, keyword):
        violations = self.history(tmp_path, f"{comment}\n")
        assert [v.message for v in violations] == [f"à relire : mot d'historique « {keyword} »"]
        assert not violations[0].blocking
        assert violations[0].context == comment.lstrip("# ")

    def test_docstring_is_scanned(self, tmp_path):
        violations = self.history(tmp_path, 'def f():\n    """Return the value, formerly a list."""\n')
        assert len(violations) == 1 and violations[0].line == 2

    def test_string_literal_is_not_scanned(self, tmp_path):
        assert self.history(tmp_path, 'message = "auparavant et désormais"\n') == []


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------


class TestHeader:
    def header(self, tmp_path, body):
        return cc.check_python_header(write(tmp_path, "mod.py", body), tmp_path)

    def test_valid_header_passes(self, tmp_path):
        assert self.header(tmp_path, GOOD_HEADER + "x = 1\n") == []

    def test_missing_header(self, tmp_path):
        violations = self.header(tmp_path, "x = 1\n")
        assert len(violations) == 1 and "absente" in violations[0].message

    def test_empty_file(self, tmp_path):
        assert len(self.header(tmp_path, "")) == 1

    def test_header_not_first_statement(self, tmp_path):
        assert len(self.header(tmp_path, "import os\n" + GOOD_HEADER)) == 1

    def test_too_short(self, tmp_path):
        violations = self.header(tmp_path, '"""Too short."""\n')
        assert len(violations) == 1 and "trop court" in violations[0].message

    def test_accented_header_is_not_english(self, tmp_path):
        violations = self.header(tmp_path, '"""Fonctions utilitaires déjà décrites en anglais ici."""\n')
        assert len(violations) == 1 and "non anglais" in violations[0].message

    def test_french_word_header_is_not_english(self, tmp_path):
        violations = self.header(tmp_path, '"""Tools for the tests dans le package of the checker."""\n')
        assert len(violations) == 1 and "non anglais" in violations[0].message

    def test_header_with_reference(self, tmp_path):
        violations = self.header(tmp_path, '"""Tests of the loader, as decided in section 4 of the notes."""\n')
        assert len(violations) == 1 and "référence" in violations[0].message

    def test_statement_before_docstring_is_not_a_header(self, tmp_path):
        assert len(self.header(tmp_path, "x = 1\n" + GOOD_HEADER)) == 1

    def test_unparsable_file(self, tmp_path):
        assert "illisible" in self.header(tmp_path, "def (:\n")[0].message


class TestConfigHeader:
    def config(self, tmp_path, name, text):
        return cc.check_config_header(write(tmp_path, name, text), tmp_path)

    def test_yaml_with_comment_passes(self, tmp_path):
        assert self.config(tmp_path, "a.yml", "# CI configuration.\nname: x\n") == []

    def test_blank_lines_before_comment_are_tolerated(self, tmp_path):
        assert self.config(tmp_path, "a.yaml", "\n\n# Header.\nname: x\n") == []

    def test_yaml_without_comment_fails(self, tmp_path):
        violations = self.config(tmp_path, "a.yml", "name: x\n# later\n")
        assert len(violations) == 1 and violations[0].line == 1

    def test_shebang_and_language_server_are_tolerated(self, tmp_path):
        text = "#!/usr/bin/env x\n# yaml-language-server: $schema=y\n# Header.\nname: x\n"
        assert self.config(tmp_path, "a.yml", text) == []

    def test_language_server_alone_is_not_a_header(self, tmp_path):
        text = "# yaml-language-server: $schema=y\nname: x\n"
        assert len(self.config(tmp_path, "a.yml", text)) == 1

    def test_toml(self, tmp_path):
        assert self.config(tmp_path, "p.toml", "# Project metadata.\n[tool]\n") == []
        assert len(self.config(tmp_path, "q.toml", "[tool]\n")) == 1

    def test_empty_file(self, tmp_path):
        assert len(self.config(tmp_path, "e.yml", "\n")) == 1


# ---------------------------------------------------------------------------
# length
# ---------------------------------------------------------------------------


class TestLength:
    @staticmethod
    def lines(count):
        return "x = 1\n" * count

    def test_test_file_at_limit_passes(self, tmp_path):
        path = write(tmp_path, "tests/unit/test_a.py", self.lines(1200))
        assert cc.check_length_file(path, tmp_path) == []

    def test_test_file_above_limit_blocks(self, tmp_path):
        path = write(tmp_path, "tests/unit/test_a.py", self.lines(1201))
        violations = cc.check_length_file(path, tmp_path)
        assert len(violations) == 1 and violations[0].blocking

    def test_package_file_above_limit_is_informative(self, tmp_path):
        path = write(tmp_path, "tsforecast/big.py", self.lines(1501))
        violations = cc.check_length_file(path, tmp_path)
        assert len(violations) == 1 and not violations[0].blocking

    def test_package_file_at_limit_passes(self, tmp_path):
        path = write(tmp_path, "tsforecast/big.py", self.lines(1500))
        assert cc.check_length_file(path, tmp_path) == []

    def test_long_script_is_ignored(self, tmp_path):
        path = write(tmp_path, "scripts/big.py", self.lines(2000))
        assert cc.check_length_file(path, tmp_path) == []


# ---------------------------------------------------------------------------
# doctest
# ---------------------------------------------------------------------------


class TestDoctest:
    def test_passing_and_failing_doctests(self, tmp_path):
        write(tmp_path, "pkg/__init__.py", GOOD_HEADER)
        write(tmp_path, "pkg/good.py", GOOD_HEADER + 'def f():\n    """Doc.\n\n    >>> 1 + 1\n    2\n    """\n')
        write(tmp_path, "pkg/bad.py", GOOD_HEADER + 'def g():\n    """Doc.\n\n    >>> 1 + 1\n    3\n    """\n')
        violations = cc.run_doctests(tmp_path, ["pkg"])
        assert [v.path for v in violations] == ["pkg/bad.py"]
        assert violations[0].rule == "doctest" and violations[0].blocking

    def test_no_doctest_is_not_an_error(self, tmp_path):
        write(tmp_path, "pkg/__init__.py", GOOD_HEADER)
        assert cc.run_doctests(tmp_path, ["pkg"]) == []

    def test_import_error_is_reported(self, tmp_path):
        write(tmp_path, "pkg/__init__.py", GOOD_HEADER)
        write(tmp_path, "pkg/broken.py", GOOD_HEADER + "import module_that_does_not_exist\n")
        violations = cc.run_doctests(tmp_path, ["pkg"])
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# collecte, résumé, point d'entrée
# ---------------------------------------------------------------------------


class TestCollection:
    def test_exclusions(self, tmp_path):
        for relative in [
            "tsforecast/a.py",
            "tsforecast/__pycache__/b.py",
            "tests/support/golden/c.py",
            "tests/support/d.py",
            ".venv/e.py",
            "site/f.yml",
            "htmlcov/g.yml",
            "docs/h.yml",
        ]:
            write(tmp_path, relative, "x = 1\n")
        found = cc.collect_files(tmp_path, [tmp_path], frozenset({".py", ".yml"}))
        names = [cc.relative_path(p, tmp_path) for p in found]
        assert names == ["docs/h.yml", "tests/support/d.py", "tsforecast/a.py"]

    def test_explicit_file_target(self, tmp_path):
        path = write(tmp_path, "a.py", "x = 1\n")
        write(tmp_path, "b.txt", "x")
        assert cc.collect_files(tmp_path, [path], frozenset({".py"})) == [path.resolve()]

    @pytest.mark.parametrize(
        "relative, expected",
        [
            ("tsforecast/frequency/a.py", "tsforecast/frequency"),
            ("tsforecast/utils/time/a.py", "tsforecast/utils"),
            ("tests/unit/frequency/test_a.py", "tests/unit/frequency"),
            ("tests/unit/utils/time/test_a.py", "tests/unit/utils"),
            ("a.py", "."),
        ],
    )
    def test_directory_key(self, relative, expected):
        assert cc.directory_key(relative) == expected


class TestMain:
    def make_repo(self, root: Path, bad: bool) -> None:
        write(root, "tsforecast/__init__.py", GOOD_HEADER)
        body = GOOD_HEADER + ("# voir § 3\n" if bad else "# Calcul exact\n")
        write(root, "tsforecast/mod.py", body)

    def test_exit_zero_when_clean(self, tmp_path, capsys):
        self.make_repo(tmp_path, bad=False)
        code = cc.main(["--root", str(tmp_path), "--rule", "refs"])
        assert code == 0
        assert "TOTAL" in capsys.readouterr().out

    def test_exit_one_on_blocking_violation(self, tmp_path, capsys):
        self.make_repo(tmp_path, bad=True)
        assert cc.main(["--root", str(tmp_path), "--rule", "refs"]) == 1
        assert "Violations bloquantes : 1" in capsys.readouterr().out

    def test_history_alone_never_blocks(self, tmp_path):
        write(tmp_path, "tsforecast/mod.py", GOOD_HEADER + "# auparavant\n")
        assert cc.main(["--root", str(tmp_path), "--rule", "history"]) == 0

    def test_detail_format_lists_violations(self, tmp_path, capsys):
        self.make_repo(tmp_path, bad=True)
        cc.main(["--root", str(tmp_path), "--rule", "refs", "--format", "detail"])
        out = capsys.readouterr().out
        assert "tsforecast/mod.py:2: [refs]" in out

    def test_explicit_path_restricts_scope(self, tmp_path):
        self.make_repo(tmp_path, bad=True)
        clean = tmp_path / "tsforecast" / "__init__.py"
        assert cc.main(["--root", str(tmp_path), "--rule", "refs", str(clean)]) == 0

    def test_header_rule_covers_config_files(self, tmp_path, capsys):
        self.make_repo(tmp_path, bad=False)
        write(tmp_path, "ci.yml", "name: x\n")
        assert cc.main(["--root", str(tmp_path), "--rule", "header", "--format", "detail"]) == 1
        assert "ci.yml:1" in capsys.readouterr().out

    def test_informative_length_listed_in_summary(self, tmp_path, capsys):
        write(tmp_path, "tsforecast/big.py", GOOD_HEADER + "x = 1\n" * 1500)
        assert cc.main(["--root", str(tmp_path), "--rule", "length"]) == 0
        assert "tsforecast/big.py" in capsys.readouterr().out


class TestSummary:
    def test_counts_per_directory_and_rule(self):
        violations = [
            cc.Violation("refs", "tsforecast/a/x.py", 1, "m"),
            cc.Violation("refs", "tsforecast/a/y.py", 2, "m"),
            cc.Violation("history", "tests/unit/a/t.py", 3, "m", blocking=False),
        ]
        table = cc.format_summary(violations, ["refs", "history"])
        lines = table.splitlines()
        assert lines[0].split()[:2] == ["dossier", "refs"]
        rows = {line.split()[0]: line.split()[1:] for line in lines[2:]}
        assert rows["tsforecast/a"] == ["2", "0", "2"]
        assert rows["tests/unit/a"] == ["0", "1", "1"]
        assert lines[-1].split() == ["TOTAL", "2", "1", "3"]


# ---------------------------------------------------------------------------
# --verify-code-unchanged
# ---------------------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    """Run git in ``root`` with a constant identity, failing the test on error."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git n'est pas disponible dans cet environnement de test")
class TestVerifyCodeUnchanged:
    ORIGINAL = GOOD_HEADER + 'def f(x):\n    """Add one."""\n    return x + 1\n\n\nclass A:\n    """Doc."""\n'

    @pytest.fixture
    def repo(self, tmp_path):
        git(tmp_path, "init", "-q")
        write(tmp_path, "tsforecast/mod.py", self.ORIGINAL)
        git(tmp_path, "add", "-A")
        git(tmp_path, "commit", "-q", "-m", "init")
        return tmp_path

    def verify(self, repo):
        scope = cc.collect_files(repo, [repo / "tsforecast"], frozenset({".py"}))
        return cc.verify_code_unchanged(repo, "HEAD", scope)

    def test_unmodified_repo(self, repo):
        assert self.verify(repo) == []

    def test_docstring_and_comment_changes_are_accepted(self, repo):
        text = self.ORIGINAL.replace("Add one.", "Increment the value.\n\n    Args:\n        x: value.")
        text = text.replace("    return x + 1", "    # Ajout d'une unité\n    return x + 1")
        text = text.replace('class A:\n    """Doc."""\n', 'class A:\n    """Another doc."""\n')
        write(repo, "tsforecast/mod.py", text)
        assert self.verify(repo) == []

    def test_docstring_added_to_docstring_less_function(self, repo):
        write(repo, "tsforecast/mod.py", self.ORIGINAL.replace('    """Add one."""\n', ""))
        git(repo, "commit", "-q", "-am", "strip")
        write(repo, "tsforecast/mod.py", self.ORIGINAL)
        assert self.verify(repo) == []

    def test_code_change_is_reported_with_diff(self, repo):
        write(repo, "tsforecast/mod.py", self.ORIGINAL.replace("x + 1", "x + 2"))
        violations = self.verify(repo)
        assert len(violations) == 1
        assert violations[0].rule == "verify" and violations[0].blocking
        assert "-" in violations[0].context and "+" in violations[0].context

    def test_created_file_is_ignored(self, repo):
        write(repo, "tsforecast/new.py", GOOD_HEADER + "x = 1\n")
        assert self.verify(repo) == []

    def test_deleted_file_is_ignored(self, repo):
        (repo / "tsforecast" / "mod.py").unlink()
        assert self.verify(repo) == []

    def test_main_runs_only_verification_when_alone(self, repo, capsys):
        write(repo, "tsforecast/mod.py", self.ORIGINAL.replace("x + 1", "x + 2"))
        assert cc.main(["--root", str(repo), "--verify-code-unchanged"]) == 1
        out = capsys.readouterr().out
        assert "verify" in out and "refs" not in out

    def test_bad_ref_reports_git_failure(self, repo):
        scope = cc.collect_files(repo, [repo / "tsforecast"], frozenset({".py"}))
        violations = cc.verify_code_unchanged(repo, "no-such-ref", scope)
        assert len(violations) == 1 and "git a échoué" in violations[0].message
