"""A throwaway expert screen on synthetic documents, for UI work.

Builds the synthetic batch of `tests/review_fixture.py` (invented rulings, no real
document or label) in a scratch root and serves the loop's open-access screen on a
spare port. Never touches the live screen (5056) or the loop's database.

    PYTHONPATH=src python scripts/dev_review_screen.py --root <empty dir> --port 5057
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from review_fixture import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=5057)
    args = parser.parse_args()
    if args.port == 5056:
        raise SystemExit("5056 is the live expert screen")
    args.root.mkdir(parents=True, exist_ok=True)
    if any(args.root.iterdir()):
        raise SystemExit(f"{args.root} is not empty; give a fresh scratch directory")
    _, app = build(args.root)
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    print(f"dev review screen: http://127.0.0.1:{args.port}/review/next", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
