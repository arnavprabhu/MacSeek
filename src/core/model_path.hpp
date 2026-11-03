#pragma once
// Where the model GGUF lives. Default model: ISTA-DASLab DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder (IQ2_XXS build,
// 192 experts per layer; split 6 = the Engram file, see gguf.cpp). The previous Unsloth UD-IQ1_S build still works.
//   MACSEEK_MODEL      full path to split 00001 (wins; handled by callers)
//   MACSEEK_MODEL_DIR  directory holding the split files (Coder: the IQ2_XXS/ folder; Unsloth: the UD-IQ1_S/ folder)
//   otherwise, in order: <source tree>/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS, the Hugging Face
//   hub cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub) repo
//   ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF, then the Unsloth repo (pinned snapshot first).
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <string>

namespace ms {

inline constexpr const char* kCoderSplit1 = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf";
inline constexpr const char* kUnslothSplit1 = "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf";
inline constexpr const char* kModelSplit1 = kCoderSplit1;

inline std::string hf_hub_dir() {
  if (const char* e = getenv("HF_HUB_CACHE"); e && *e) return e;
  if (const char* e = getenv("HF_HOME"); e && *e) return std::string(e) + "/hub";
  const char* home = getenv("HOME");
  return std::string(home ? home : "") + "/.cache/huggingface/hub";
}

inline bool model_file_exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

// first snapshots/<sha>/<sub> of an HF repo dir that holds `split1` ("" if none)
inline std::string hf_snapshot_with(const std::string& repo_dir, const char* sub, const char* split1,
                                    const char* pinned = nullptr) {
  const std::string snaps = hf_hub_dir() + "/" + repo_dir + "/snapshots/";
  if (pinned) {
    std::string d = snaps + pinned + "/" + sub;
    if (model_file_exists(d + "/" + split1)) return d;
  }
  std::string found;
  if (DIR* dir = opendir(snaps.c_str())) {
    while (dirent* ent = readdir(dir)) {
      if (ent->d_name[0] == '.') continue;
      std::string d = snaps + ent->d_name + "/" + sub;
      if (model_file_exists(d + "/" + split1)) { found = d; break; }
    }
    closedir(dir);
  }
  return found;
}

// full path of split 00001
inline std::string default_model_split1() {
  if (const char* e = getenv("MACSEEK_MODEL_DIR"); e && *e) {
    const std::string d = e;
    if (!model_file_exists(d + "/" + kCoderSplit1) && model_file_exists(d + "/" + kUnslothSplit1)) return d + "/" + kUnslothSplit1;
    return d + "/" + kCoderSplit1;
  }
#ifdef MACSEEK_SOURCE_DIR
  {
    const std::string d = std::string(MACSEEK_SOURCE_DIR) + "/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS";
    if (model_file_exists(d + "/" + kCoderSplit1)) return d + "/" + kCoderSplit1;
  }
#endif
  std::string d = hf_snapshot_with("models--ISTA-DASLab--DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF", "IQ2_XXS", kCoderSplit1);
  if (!d.empty()) return d + "/" + kCoderSplit1;
  d = hf_snapshot_with("models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF", "UD-IQ1_S", kUnslothSplit1,
                       "8a6ec9cf118e785a362ea94ee123c61c70553be6");
  if (!d.empty()) return d + "/" + kUnslothSplit1;
#ifdef MACSEEK_SOURCE_DIR
  return std::string(MACSEEK_SOURCE_DIR) + "/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/" + kCoderSplit1;
#else
  return hf_hub_dir() + "/models--ISTA-DASLab--DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/" + kCoderSplit1;
#endif
}

inline std::string default_model_dir() {
  const std::string p = default_model_split1();
  return p.substr(0, p.rfind('/'));
}

}  // namespace ms
