"""
Precompute PCA projection maps from validation-set embeddings.

Usage (from repo root):
  python -m ml.training.build_projection_maps
  python -m ml.training.build_projection_maps --demo
  python -m ml.training.build_projection_maps --lang en --model bert
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ml.visualization.embedding_projection import build_projection_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Build PCA scatter-plot projection artifacts.")
    parser.add_argument("--out-dir", type=Path, default=root / "models" / "projections")
    parser.add_argument("--lang", choices=["en", "pl", "all"], default="all")
    parser.add_argument("--model", choices=["tfidf_lr", "bert", "all"], default="all")
    parser.add_argument("--data-en", type=Path, default=root / "data" / "raw" / "train.csv")
    parser.add_argument("--data-pl", type=Path, default=root / "BAN-PL_2" / "BAN-PL.csv")
    parser.add_argument("--demo", action="store_true", help="Use tiny demo corpus when CSV is missing.")
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--max-embed", type=int, default=2500, help="Max hold-out rows to embed (BERT speed)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    combos: list[tuple[str, str, Path]] = []
    if args.lang in ("en", "all"):
        combos.append(("en", "tfidf_lr", root / "models" / "model.pkl"))
        combos.append(("en", "bert", root / "models" / "bert"))
    if args.lang in ("pl", "all"):
        combos.append(("pl", "tfidf_lr", root / "models" / "model_pl.pkl"))
        combos.append(("pl", "bert", root / "models" / "bert_pl"))

    if args.model != "all":
        combos = [c for c in combos if c[1] == args.model]

    manifest_entries: list[dict] = []
    for lang, model_kind, artifact in combos:
        if not artifact.exists():
            print(f"Skip {lang}/{model_kind}: artifact missing at {artifact}", file=sys.stderr)
            continue
        data_path = args.data_en if lang == "en" else args.data_pl
        use_demo = args.demo or not data_path.is_file()
        if use_demo:
            print(f"Building {lang}/{model_kind} with demo/fallback corpus ...", flush=True)
        else:
            print(f"Building {lang}/{model_kind} from {data_path} ...", flush=True)

        out_dir = args.out_dir / lang / model_kind
        try:
            meta = build_projection_bundle(
                lang=lang,  # type: ignore[arg-type]
                model_kind=model_kind,  # type: ignore[arg-type]
                model_artifact=artifact,
                out_dir=out_dir,
                data_path=data_path,
                demo=use_demo,
                random_state=args.random_state,
                max_points=args.max_points,
                max_embed=args.max_embed,
            )
            manifest_entries.append(meta)
            print(
                f"  OK: {meta['n_displayed']} points / {meta['n_total_test']} test "
                f"({meta['method']}) → {out_dir}",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAILED {lang}/{model_kind}: {exc}", file=sys.stderr)

    if not manifest_entries:
        print("No projection artifacts were built.", file=sys.stderr)
        return 1

    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "random_state": args.random_state,
        "entries": manifest_entries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote manifest to {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
