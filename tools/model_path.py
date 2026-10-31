#!/usr/bin/env python3
"""Default location of the DeepSeek-V4.1-Flash-1017 UD-IQ1_S GGUF (split 00001); same rules as src/core/model_path.hpp.

  MACSEEK_MODEL      full path to split 00001 (wins if set)
  MACSEEK_MODEL_DIR  directory holding the 5 UD-IQ1_S splits
  otherwise the Hugging Face hub cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).

Run it directly to print the resolved path (used by the shell tools).
"""
import glob
import os

REPO = "models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF"
PINNED = "8a6ec9cf118e785a362ea94ee123c61c70553be6"
SPLIT1 = "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"


def hf_hub_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def default_model_dir():
    if os.environ.get("MACSEEK_MODEL_DIR"):
        return os.environ["MACSEEK_MODEL_DIR"]
    snaps = os.path.join(hf_hub_dir(), REPO, "snapshots")
    pinned = os.path.join(snaps, PINNED, "UD-IQ1_S")
    if os.path.exists(os.path.join(pinned, SPLIT1)):
        return pinned
    for d in sorted(glob.glob(os.path.join(snaps, "*", "UD-IQ1_S"))):
        if os.path.exists(os.path.join(d, SPLIT1)):
            return d
    return pinned


def default_model():
    return os.environ.get("MACSEEK_MODEL") or os.path.join(default_model_dir(), SPLIT1)


if __name__ == "__main__":
    print(default_model())
