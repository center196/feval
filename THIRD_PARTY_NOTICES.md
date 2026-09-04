# Third-party dataset notices

Feval builds every evaluation window from immutable public dataset revisions.
It redistributes no dataset content; each miner and validator fetches the rows
it needs directly from Hugging Face at the pinned commit. This file records the
attribution those licences require, and the exact revisions in use.

Licence texts are summarised here for orientation. The authoritative terms are
the ones published on each dataset page.

## Evaluation sources

| Source | Revision | Licence | Rows per window |
| --- | --- | --- | --- |
| [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | `d3d08664755704f422af97d43a7ff0ded4bd95df` | CC BY 4.0 | 3,000 |
| [nvidia/Nemotron-CrossThink](https://huggingface.co/datasets/nvidia/Nemotron-CrossThink) | `a4ce9a3b9434c5f231e2cbe30696d9a721c11d69` | CC BY 4.0 | 1,500 |
| [nvidia/Nemotron-RL-knowledge-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa) | `62a1eec1f952723eab2ee3832222f533b8138067` | CC BY 4.0 | 2,000 |
| [nvidia/OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) | `174b02c9cdf231f220765b2a1d5ece4550921894` | CC BY 4.0 | 1,500 |
| [PrimeIntellect/synthetic-code-understanding](https://huggingface.co/datasets/PrimeIntellect/synthetic-code-understanding) | `106a1cec075ae29b8dc07e355a29ddce2cf0745b` | Apache-2.0, see note | 2,000 |

All five are published for commercial use.

### Attribution

- **OpenMathReasoning**, **Nemotron-CrossThink**, **Nemotron-RL-knowledge-mcqa**
  and **OpenScienceReasoning-2** are © NVIDIA Corporation, released under the
  [Creative Commons Attribution 4.0 International Licence](https://creativecommons.org/licenses/by/4.0/legalcode).
- **Nemotron-CrossThink** additionally notes that a model trained on it may be
  subject to the redistribution and use terms of the
  [Qwen Licence Agreement](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE).
- **Nemotron-RL-knowledge-mcqa** combines and refines subsets of
  OpenScienceReasoning-2 together with other unstructured sources.

### Note on `PrimeIntellect/synthetic-code-understanding`

The subset repository carries no licence tag of its own. Its licence comes from
the parent release, [PrimeIntellect/SYNTHETIC-1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1),
which declares **Apache-2.0** and which names this repository as the origin of
its `code_output_prediction` rows in its own `hf_dataset_name` column. The
subset's card likewise describes itself as "a subset of the task data used to
construct SYNTHETIC-1".

If you need a licence grant attached to the artefact you actually read, take the
`code_output_prediction` rows from `PrimeIntellect/SYNTHETIC-1` instead, where
the grant is explicit.

## Retired sources

Windows built before protocol `feval-network-v34` drew on two further datasets.
They are no longer read, and are listed here because earlier evaluation
artefacts derived from them may still exist:

| Source | Licence |
| --- | --- |
| [nvidia/Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4) | mixed: CC BY-SA 4.0 (Math StackExchange, ~64%) and CC BY 4.0 (AoPS, ~36%) |
| [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | ODC-BY |

`Nemotron-SFT-Math-v4` carries a per-row `license` column because its rows are
not uniformly licensed. Its Math StackExchange majority is **CC BY-SA 4.0**,
which adds a ShareAlike condition that CC BY does not. Anyone redistributing an
evaluation file generated from that dataset under the older protocol should
carry that condition forward.

## What Feval does with this data

Rows are read, filtered to those whose answer can be settled by exact string
comparison, and reduced to a question plus an expected value. No dataset code is
downloaded or executed, and no model is used to grade. Generated evaluation
files stay local to each node.
