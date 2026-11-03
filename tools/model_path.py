#!/usr/bin/env python3
"""Default location of the model GGUF (split 00001); same rules as src/core/model_path.hpp.

Default model: ISTA-DASLab DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder (IQ2_XXS build, 6 splits; split 6 = the Engram
file). The Unsloth UD-IQ1_S build is still found as a fallback.

  MACSEEK_MODEL      full path to split 00001 (wins if set)
  MACSEEK_MODEL_DIR  directory holding the splits
  otherwise <repo>/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS, then the Hugging Face hub cache
  ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).

Run it directly to print the resolved path (used by the shell tools).
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODER_REPO = "models--ISTA-DASLab--DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF"
CODER_SPLIT1 = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf"
CODER_LOCAL = os.path.join(ROOT, "models", "DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF", "IQ2_XXS")
UNSLOTH_REPO = "models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF"
UNSLOTH_PINNED = "8a6ec9cf118e785a362ea94ee123c61c70553be6"
UNSLOTH_SPLIT1 = "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"
SPLIT1 = CODER_SPLIT1


def hf_hub_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def _snapshot(repo, sub, split1, pinned=None):
    snaps = os.path.join(hf_hub_dir(), repo, "snapshots")
    if pinned and os.path.exists(os.path.join(snaps, pinned, sub, split1)):
        return os.path.join(snaps, pinned, sub)
    for d in sorted(glob.glob(os.path.join(snaps, "*", sub))):
        if os.path.exists(os.path.join(d, split1)):
            return d
    return None


def default_model():
    if os.environ.get("MACSEEK_MODEL"):
        return os.environ["MACSEEK_MODEL"]
    d = os.environ.get("MACSEEK_MODEL_DIR")
    if d:
        if not os.path.exists(os.path.join(d, CODER_SPLIT1)) and os.path.exists(os.path.join(d, UNSLOTH_SPLIT1)):
            return os.path.join(d, UNSLOTH_SPLIT1)
        return os.path.join(d, CODER_SPLIT1)
    if os.path.exists(os.path.join(CODER_LOCAL, CODER_SPLIT1)):
        return os.path.join(CODER_LOCAL, CODER_SPLIT1)
    d = _snapshot(CODER_REPO, "IQ2_XXS", CODER_SPLIT1)
    if d:
        return os.path.join(d, CODER_SPLIT1)
    d = _snapshot(UNSLOTH_REPO, "UD-IQ1_S", UNSLOTH_SPLIT1, UNSLOTH_PINNED)
    if d:
        return os.path.join(d, UNSLOTH_SPLIT1)
    return os.path.join(CODER_LOCAL, CODER_SPLIT1)


def default_model_dir():
    return os.path.dirname(default_model())


if __name__ == "__main__":
    print(default_model())
