"""Model loading + per-track inference core. Configured entirely via env vars
(schema.ENV_*, exported by the flake devShell and wrappers — store paths).

Single MAEST pass, output layer PartitionedCall/Identity_12 for BOTH artifacts:
the genre_discogs519 head was trained on layer 12 (head-faithful activations);
MTG recommends layer 7 for similarity embeddings — recorded per-file so a
layer-7 re-extract remains possible later (GPU-only, non-blocking).

Verified pipeline facts (project memory: pipeline-facts.md):
  TensorflowPredictMAEST(graphFilename=EMB, output='PartitionedCall/Identity_12')
  TensorflowPredict(graphFilename=HEAD, inputs=['embeddings'],
                    outputs=['PartitionedCall/Identity_1'])   # sigmoid probs
  head input is (batch, 1685, 768) via es.Pool; TensorflowPredict2D CANNOT run it.
  Patches: 1876 mel frames window (~30.02s), 1875 hop; pad <30.5s tracks with
  silence (MAEST errors "input signal is too short" below ~30.02s).
"""
import base64
import json
import os

import numpy as np

import schema

EMB_LAYER = 'PartitionedCall/Identity_12'
HEAD_INPUT = 'embeddings'
HEAD_OUT = 'PartitionedCall/Identity_1'  # sigmoid probabilities
EMB_MODEL_NAME = 'discogs-maest-30s-pw-519l-2.pb'
HEAD_MODEL_NAME = 'genre_discogs519-discogs-maest-30s-pw-519l-1.pb'


def model_paths():
    emb = os.environ.get(schema.ENV_EMBEDDING_MODEL)
    head = os.environ.get(schema.ENV_CLASSIFIER_HEAD)
    classes_json = os.environ.get(schema.ENV_CLASSES_JSON)
    missing = [k for k, v in [
        (schema.ENV_EMBEDDING_MODEL, emb),
        (schema.ENV_CLASSIFIER_HEAD, head),
        (schema.ENV_CLASSES_JSON, classes_json),
    ] if not v]
    if missing:
        raise SystemExit(f'missing env vars: {", ".join(missing)} '
                         '(enter the flake devShell or run via a flake wrapper)')
    return emb, head, classes_json


def load_classes(classes_json):
    with open(classes_json) as f:
        return json.load(f)['classes']


def models_record():
    """Per-file provenance block (Contract A 'models'). Bare filenames, not
    store paths — the same track JSON must stay comparable across hosts."""
    return {
        'embedding': EMB_MODEL_NAME,
        'embedding_layer': EMB_LAYER,
        'head': HEAD_MODEL_NAME,
        'head_output': HEAD_OUT,
    }


class Classifier:
    """Lazily-initialized inference engine (heavy: loads both graphs + TF session)."""

    def __init__(self):
        from essentia.standard import TensorflowPredictMAEST, TensorflowPredict
        from essentia import Pool
        emb_path, head_path, classes_json = model_paths()
        self._Pool = Pool
        self.classes = load_classes(classes_json)
        self._emb = TensorflowPredictMAEST(graphFilename=emb_path, output=EMB_LAYER)
        self._head = TensorflowPredict(graphFilename=head_path,
                                       inputs=[HEAD_INPUT], outputs=[HEAD_OUT])

    def classify(self, audio: np.ndarray):
        """audio: float32 mono 16kHz (padded to >= schema.MIN_SECONDS by caller
        if short). Returns (pooled_embedding [768], activations {class: float})."""
        emb = np.asarray(self._emb(audio))                  # (n_patches, 1, 1685, 768)
        pooled_emb = emb.reshape(-1, schema.EMB_DIM).mean(axis=0)

        pool = self._Pool()
        pool.set(HEAD_INPUT, emb)
        preds = self._head(pool)[HEAD_OUT]                  # (n, 1, 1, 519)
        acts_mean = np.asarray(preds).reshape(-1, schema.N_CLASSES).mean(axis=0)

        activations = {c: round(float(a), 6) for c, a in zip(self.classes, acts_mean)}
        return pooled_emb, activations


def encode_embedding(vec: np.ndarray, layer: str = EMB_LAYER) -> dict:
    """float32 vector -> Contract-A embedding block (base64, with provenance)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return {
        'dtype': 'float32',
        'shape': list(v.shape),
        'layer': layer,
        'data': base64.b64encode(v.tobytes()).decode('ascii'),
    }
