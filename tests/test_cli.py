"""Tests for the CLI commands."""

import pytest
from click.testing import CliRunner
from remesher.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_presets_command(self, runner):
        result = runner.invoke(cli, ["presets"])
        assert result.exit_code == 0
        assert "web" in result.output
        assert "mobile" in result.output
        assert "UV methods" in result.output

    def test_single_missing_file(self, runner):
        result = runner.invoke(cli, ["single", "nonexistent.glb", "out.glb"])
        assert result.exit_code != 0

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "single" in result.output
        assert "batch" in result.output

    def test_single_help(self, runner):
        result = runner.invoke(cli, ["single", "--help"])
        assert result.exit_code == 0
        assert "--uv" in result.output
        assert "xatlas" in result.output
        assert "lscm" in result.output
        assert "arap" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output
