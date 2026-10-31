#pragma once
// Where the DeepSeek-V4.1-Flash-1017 UD-IQ1_S GGUF lives.
//   MACSEEK_MODEL_DIR  directory holding the 5 UD-IQ1_S split files (wins if set)
//   otherwise the Hugging Face hub cache: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub,
//   repo unsloth/DeepSeek-V4.1-Flash-1017-GGUF, pinned snapshot first, then any other snapshot that has the files.
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <string>

namespace ms {

inline constexpr const char* kModelSplit1 = "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf";

inline std::string hf_hub_dir() {
  if (const char* e = getenv("HF_HUB_CACHE"); e && *e) return e;
  if (const char* e = getenv("HF_HOME"); e && *e) return std::string(e) + "/hub";
  const char* home = getenv("HOME");
  return std::string(home ? home : "") + "/.cache/huggingface/hub";
}

inline std::string default_model_dir() {
  if (const char* e = getenv("MACSEEK_MODEL_DIR"); e && *e) return e;
  const std::string snaps = hf_hub_dir() + "/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/";
  const std::string pinned = snaps + "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S";
  struct stat st;
  auto has = [&](const std::string& d) { return stat((d + "/" + kModelSplit1).c_str(), &st) == 0; };
  if (has(pinned)) return pinned;
  if (DIR* dir = opendir(snaps.c_str())) {
    while (dirent* ent = readdir(dir)) {
      if (ent->d_name[0] == '.') continue;
      std::string d = snaps + ent->d_name + "/UD-IQ1_S";
      if (has(d)) { closedir(dir); return d; }
    }
    closedir(dir);
  }
  return pinned;  // not found: callers report the missing file
}

inline std::string default_model_split1() { return default_model_dir() + "/" + kModelSplit1; }

}  // namespace ms
