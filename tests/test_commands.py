"""Command handlers, exercised directly against light stand-ins for their args/config.

`loop_status` only reads `config.public_root` (via `rounds_state_dir`) and
`args.round_index`, so a `SimpleNamespace`/`argparse.Namespace` stub is enough
to exercise it without building a full `AppConfig`.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.commands import loop_status


def test_loop_status_returns_zero_and_prints_json_naming_the_round(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = Namespace(round_index=3)
    config = SimpleNamespace(public_root=tmp_path)

    exit_code = loop_status(args, config)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["round"] == 3
