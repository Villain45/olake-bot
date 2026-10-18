"""Bound CPU inference while reusing FlashRank's tokenizer and scoring model."""

import logging
from pathlib import Path
from typing import Any

import onnxruntime as ort
from flashrank import Ranker, RerankRequest
from flashrank.Config import default_cache_dir, default_model, listwise_rankers, model_file_map


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class CpuRanker(Ranker):
    """FlashRank 0.2.10 with a single, explicitly configured ONNX session.

    Model preparation, tokenization, and reranking stay in FlashRank. Its
    constructor cannot accept session options, so only ONNX initialization is
    reproduced here. Optional listwise models retain FlashRank's initialization.
    """

    def __init__(
        self,
        *,
        model_name: str = default_model,
        cache_dir: str = default_cache_dir,
        threads: int = 1,
        max_length: int = 512,
    ):
        _positive_integer(threads, "threads")
        _positive_integer(max_length, "max_length")
        if model_name in listwise_rankers:
            super().__init__(model_name=model_name, cache_dir=cache_dir, max_length=max_length)
            return

        self.logger = logging.getLogger("flashrank.Ranker")
        self.cache_dir = Path(cache_dir)
        self.model_dir = self.cache_dir / model_name
        self._prepare_model_dir(model_name)
        self.llm_model = None

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(self.model_dir / model_file_map[model_name]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = self._get_tokenizer(max_length)


def rerank_batched(
    ranker: Ranker,
    query: str,
    passages: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score small batches and merge by score, preserving input order for ties.

    FlashRank mutates and sorts request dictionaries, so requests receive copies.
    Quantized scores can vary with padding and batch size; all candidates are kept.
    Passage IDs remain untouched. Listwise models rank the complete request since
    their output has no independent scores that can be merged across batches.
    """
    _positive_integer(batch_size, "batch_size")
    if not passages:
        return []

    copies = [dict(passage) for passage in passages]
    if isinstance(ranker, Ranker) and ranker.llm_model is not None:
        return ranker.rerank(RerankRequest(query=query, passages=copies))

    original_positions = {id(passage): index for index, passage in enumerate(copies)}
    ranked = []
    for start in range(0, len(copies), batch_size):
        batch = copies[start:start + batch_size]
        # Real FlashRank returns the input dictionaries. ID lookup also supports
        # simple ranker doubles that return fresh dictionaries containing scores.
        positions_by_id = {passage.get("id"): start + offset for offset, passage in enumerate(batch)}
        for result in ranker.rerank(RerankRequest(query=query, passages=batch)):
            position = original_positions.get(id(result))
            if position is None:
                position = positions_by_id[result.get("id")]
            ranked.append((position, result))

    ranked.sort(key=lambda item: (-item[1]["score"], item[0]))
    return [result for _, result in ranked]
