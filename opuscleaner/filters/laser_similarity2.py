#!/usr/bin/env python3
import logging
import os
import sys
import time
from typing import List, Tuple, Iterable, TypeVar, Optional, TextIO
import argparse
import numpy as np

os.environ["TQDM_DISABLE"] = "1"
from laser_encoders import LaserEncoderPipeline
from numpy.linalg import norm
from numpy.polynomial import Polynomial
from collections import deque
from io import TextIOBase
import pycountry
from laser_encoders import download_models
# issues with logging on model downloading
download_models.logger.setLevel(logging.ERROR)

def convert_iso_639_1_to_639_2(lang_code):
    try:
        # Find the language by its ISO 639-1 code (two-letter code)
        language = pycountry.languages.get(alpha_2=lang_code)
        # Return the ISO 639-2 code (three-letter code)
        return language.alpha_3
    except AttributeError:
        # Return None if the language code is not found
        raise ValueError(f'Language code not found: {lang_code}')

def _compute_similarity(encoder_src: LaserEncoderPipeline,
                        encoder_trg: LaserEncoderPipeline,
                        batch: List[Tuple[str, str]]) -> List[float]:
    assert len(batch) > 0
    embeddings_src = encoder_src.encode_sentences([line[0] for line in batch])
    embeddings_tgt = encoder_trg.encode_sentences([line[1] for line in batch])
    # laser in fact returns np array
    return [float(sim) for sim in _cosine_sim(embeddings_src, embeddings_tgt)]


def _cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    return np.sum(emb1 * emb2, axis=-1) / (norm(emb1, axis=-1) * norm(emb2, axis=-1))


def interpolate(sample: Iterable[Tuple[int, float]], target:float) -> int:
    poly = Polynomial.fit([duration for size, duration in sample], [size for size, duration in sample], 1)
    return int(poly(target)), poly


class NullIO(TextIOBase):
    """TextIO that does nothing, as if writing to /dev/null."""
    def write(self, data:str) -> int:
        return len(data)


T = TypeVar('T')

def chunked(iterable: Iterable[T], *, chunk_size:Optional[int]=None, chunk_time:Optional[float]=None, verbose:Optional[TextIO]=NullIO()) -> Iterable[List[T]]:
    """Self-tuning batching iterator"""
    it = iter(iterable)

    # Initial set of measurements we then interpolate from
    limit_samples = iter([32, 64, 128, 256, 512, 1024])

    # Chunk size limit for the next chunk
    limit = chunk_size or next(limit_samples)

    # Measurements (limited to most recent 32)
    measurements = deque([], maxlen=32)

    while True:
        # Create a chunk
        chunk = [el for _, el in zip(range(limit), it)]

        # Did we reach the end because the last read was accidentally
        # exactly the remainder of the dataset? Or because there was no
        # input to begin with?
        if not chunk:
            return

        # Measure how long it takes before we are asked for the next chunk
        yield_time = time.monotonic()
        yield chunk
        if len(chunk) < limit:
            break
        yield_duration = time.monotonic() - yield_time
        print(f"Chunk size {limit} took {yield_duration}s", file=verbose)
        measurements.append((limit, yield_duration))

        # If we're running in dynamic mode, update the chunk size limit
        if chunk_size is None and chunk_time is not None:
            try:
                # Next limit for sampling?
                limit = next(limit_samples)
            except StopIteration:
                # No, we've run all the samples. Use previous measurements
                limit, poly = interpolate(measurements, chunk_time)
                print(f'Fitted {poly}', file=verbose)

            print(f"Setting chunk size to {limit}", file=verbose)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Filter a parallel dataset using LASER.")
    parser.add_argument("--verbose", action="store_true", help="Print tuning info")
    parser.add_argument("--batch-size", type=int, help="LASER batch size")
    parser.add_argument("--batch-latency", type=float, default=10.0, help="Tune batch size to process a batch every N seconds (defaults to 10s, ignored if --batch-size is given)")
    parser.add_argument("--src-lang", type=str, required=True, help="Two-letter source language code (ISO 639-1)")
    parser.add_argument("--tgt-lang", type=str, required=True, help="Two-letter target language code (ISO 639-1)")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, help="Minimum accepted LASER score.")
    group.add_argument("--scores", action="store_true", help="Print scores instead of lines")

    args = parser.parse_args()

    if not args.scores and args.threshold is None:
        print("Either use --threshold or --scores", file=sys.stderr)
    # issue with reusing the same cached dir without explicitly specifying model directory
    os.makedirs(f"data/laser_{args.src_lang}", exist_ok=True)
    os.makedirs(f"data/laser_{args.tgt_lang}", exist_ok=True)
    laser_encoder_src = LaserEncoderPipeline(lang=convert_iso_639_1_to_639_2(args.src_lang),
                                             model_dir=f"data/laser_{args.src_lang}")
    laser_encoder_trg = LaserEncoderPipeline(lang=convert_iso_639_1_to_639_2(args.tgt_lang),
                                             model_dir=f"data/laser_{args.tgt_lang}")

    for batch in chunked(sys.stdin, chunk_size=args.batch_size, chunk_time=args.batch_latency, verbose=sys.stderr if args.verbose else NullIO()):
        # TODO error checking of column count?
        scores = _compute_similarity(laser_encoder_src, laser_encoder_trg,
                                     [tuple(line.rstrip("\r\n").split("\t")[:2]) for line in batch])

        if args.scores:
            for score in scores:
                print(score)
        else:
            for line, score in zip(batch, scores):
                if score >= args.threshold:
                    sys.stdout.write(line)


if __name__ == "__main__":
    main()
