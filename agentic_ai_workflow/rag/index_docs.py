from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local no-key RAG index.")
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--index-path", default="data/index/local_vectors.json")
    parser.add_argument("--chunk-size", type=int, default=90)
    parser.add_argument("--overlap", type=int, default=20)
    args = parser.parse_args()

    pipeline = RagPipeline.from_directory(
        docs_dir=Path(args.docs_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        index_path=Path(args.index_path),
        rebuild_index=True,
    )
    row_count = len(pipeline.retriever.vector_store._rows)
    print(f"Indexed {row_count} chunks into {args.index_path}")


if __name__ == "__main__":
    main()
