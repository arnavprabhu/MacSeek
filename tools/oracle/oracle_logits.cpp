// oracle_logits — TEST-ONLY tool. Links stock Homebrew libllama to produce reference logits ("oracle") for
// MacSeek's correctness gate. Never part of the engine. See docs/modules/oracle.md.
//
// Input : --ids FILE (raw little-endian int32 token ids)  or  --text FILE / --prompt STR (tokenized by llama.cpp,
//         special tokens parsed, no BOS added — the chat template text carries its own BOS).
// Output: --out FILE.logits (format below) + FILE.logits.json (metadata/timing) ; --dump-ids FILE writes the ids used.
//
// .logits format v1 (little-endian):
//   char magic[4] = "MSLG"; u32 version = 1; u32 n_tokens; u32 n_vocab; u32 mode (0 = full, 1 = topk);
//   u32 k (0 for full); u32 first_pos (index of the first position with logits, normally 0); u32 n_pos (positions stored)
//   i32 tokens[n_tokens]
//   mode 0: f32 logits[n_pos][n_vocab]                      (row p = logits predicting token p+first_pos+1)
//   mode 1: per position: f32 lse; f32 pad=0; i32 ids[k]; f32 logits[k]   (ids sorted by logit desc; lse = logsumexp
//           over the full vocab, so p(id) = exp(logit - lse); tail mass = 1 - sum_k p)
//
// --list F: run many (ids, out) jobs with a single model load (KV cleared between jobs).
// Other modes: --tokenize-only (vocab-only load, no weights, no lock needed), --gen N (greedy continuation after the
// prompt, prints text; used by the needle check).
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void die(const char *msg) { fprintf(stderr, "oracle_logits: %s\n", msg); exit(1); }

static std::string read_file(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static std::string json_escape(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

static void usage() {
    fprintf(stderr,
        "usage: oracle_logits -m MODEL (--ids F | --text F | --prompt S | --list F) [options]\n"
        "  --out F           logits output (.logits); omit to skip logits\n"
        "  --dump-ids F      write the token ids used (raw int32)\n"
        "  --topk K          store top-K + logsumexp per position instead of full logits (0 = full, default)\n"
        "  --last N          only store the last N positions (default: all)\n"
        "  --gen N           greedy-generate N tokens after the prompt, print them (stderr+json)\n"
        "  --tokenize-only   load vocab only, tokenize, write --dump-ids, exit\n"
        "  --ctx N           context size (default n_tokens + gen + 16)\n"
        "  --batch N / --ubatch N   (default 2048 / 512)\n"
        "  --threads N       CPU threads (default 6 = llama-bench default)\n"
        "  --ngl N           GPU layers (default 99)\n"
        "  --cpu-moe 0|1     keep MoE expert tensors on CPU (default 1; full offload fails: Metal OOM)\n"
        "  --mmap 0|1        (default 1)\n"
        "  --fa on|off|auto  flash attention (default auto)\n"
        "  --swa-full 0|1    full-size SWA KV cache (default: llama.cpp default = 1; use 0 for 200K runs)\n"
        "  --list F          many jobs, one model load: lines \"IDS_FILE OUT_FILE\" (OUT_FILE.json gets metadata)\n"
        "  --json F          metadata json path when no --out is given (e.g. with --gen)\n"
        "  --quiet           suppress llama.cpp logging\n");
    exit(2);
}

static bool g_quiet = false;
static void log_cb(ggml_log_level level, const char *text, void *) {
    if (g_quiet && level < GGML_LOG_LEVEL_WARN) return;
    fputs(text, stderr);
}

int main(int argc, char **argv) {
    std::string model, ids_path, text_path, prompt, out_path, dump_ids, list_path, json_path;
    int topk = 0, last = -1, gen = 0, ctx = 0, nbatch = 2048, nubatch = 512, threads = 6, ngl = 99;
    bool cpu_moe = true, use_mmap = true, tok_only = false, have_prompt = false;
    int swa_full = -1;  // -1 = library default
    std::string fa = "auto";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--text") text_path = next();
        else if (a == "--prompt") { prompt = next(); have_prompt = true; }
        else if (a == "--out") out_path = next();
        else if (a == "--dump-ids") dump_ids = next();
        else if (a == "--topk") topk = atoi(next().c_str());
        else if (a == "--last") last = atoi(next().c_str());
        else if (a == "--gen") gen = atoi(next().c_str());
        else if (a == "--tokenize-only") tok_only = true;
        else if (a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "--batch") nbatch = atoi(next().c_str());
        else if (a == "--ubatch") nubatch = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--ngl") ngl = atoi(next().c_str());
        else if (a == "--cpu-moe") cpu_moe = atoi(next().c_str()) != 0;
        else if (a == "--mmap") use_mmap = atoi(next().c_str()) != 0;
        else if (a == "--fa") fa = next();
        else if (a == "--quiet") g_quiet = true;
        else if (a == "--list") list_path = next();
        else if (a == "--swa-full") swa_full = atoi(next().c_str());
        else if (a == "--json") json_path = next();
        else usage();
    }
    if (model.empty()) usage();
    int nsrc = !ids_path.empty() + !text_path.empty() + have_prompt + !list_path.empty();
    if (nsrc != 1) usage();

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    double t0 = now_s();

    // ---- model params (identical for every dump; recorded in the json) ----
    static const llama_model_tensor_buft_override overrides[] = {
        {"\\.ffn_(up|down|gate)_(ch|)exps", nullptr},  // buft filled below
        {nullptr, nullptr},
    };
    std::vector<llama_model_tensor_buft_override> ov(overrides, overrides + 2);
    ov[0].buft = ggml_backend_cpu_buffer_type();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.vocab_only = tok_only;
    if (!use_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    if (cpu_moe && !tok_only) mp.tensor_buft_overrides = ov.data();

    llama_model *mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) die("model load failed");
    double t_load = now_s() - t0;
    const llama_vocab *vocab = llama_model_get_vocab(mdl);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- tokens ----
    std::vector<llama_token> toks;
    if (!list_path.empty()) {
        // tokens come from the list file (see below)
    } else if (!ids_path.empty()) {
        std::string raw = read_file(ids_path);
        if (raw.size() % 4) die("ids file size not a multiple of 4");
        toks.resize(raw.size() / 4);
        memcpy(toks.data(), raw.data(), raw.size());
    } else {
        std::string text = have_prompt ? prompt : read_file(text_path);
        int n = -llama_tokenize(vocab, text.data(), (int)text.size(), nullptr, 0, /*add_special*/ false, /*parse_special*/ true);
        toks.resize(n);
        int n2 = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), n, false, true);
        if (n2 != n) die("tokenize size mismatch");
    }
    for (auto t : toks) if (t < 0 || t >= n_vocab) die("token id out of range");
    const int n_tok = (int)toks.size();
    fprintf(stderr, "oracle_logits: %d tokens, n_vocab %d\n", n_tok, n_vocab);
    if (!dump_ids.empty()) {
        FILE *f = fopen(dump_ids.c_str(), "wb");
        if (!f) die("cannot write dump-ids");
        fwrite(toks.data(), 4, toks.size(), f); fclose(f);
    }
    if (tok_only) { llama_model_free(mdl); return 0; }

    // ---- jobs: one (tokens, out) pair, or many via --list (model loaded once) ----
    struct Job { std::vector<llama_token> toks; std::string out; };
    std::vector<Job> jobs;
    if (!list_path.empty()) {
        std::ifstream lf(list_path);
        std::string ip, op;
        while (lf >> ip >> op) {
            std::string raw = read_file(ip);
            Job jb; jb.toks.resize(raw.size() / 4); memcpy(jb.toks.data(), raw.data(), jb.toks.size() * 4); jb.out = op;
            for (auto t : jb.toks) if (t < 0 || t >= n_vocab) die("token id out of range");
            jobs.push_back(std::move(jb));
        }
    } else {
        jobs.push_back({toks, out_path});
    }
    size_t max_tok = 0;
    for (auto &jb : jobs) max_tok = std::max(max_tok, jb.toks.size());

    // ---- context ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx > 0 ? ctx : (uint32_t)(max_tok + gen + 16);
    cp.n_batch = nbatch;
    cp.n_ubatch = nubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = false;
    if (swa_full >= 0) cp.swa_full = swa_full != 0;
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                              : LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context *lctx = llama_init_from_model(mdl, cp);
    if (!lctx) die("context init failed");
    llama_batch batch = llama_batch_init(nbatch, 0, 1);
    std::vector<int> idx(n_vocab);

  for (size_t jn = 0; jn < jobs.size(); jn++) {
    const std::vector<llama_token> &toks = jobs[jn].toks;
    const std::string &out_path = jobs[jn].out;
    const int n_tok = (int)toks.size();
    if (n_tok == 0) die("empty prompt");
    llama_memory_clear(llama_get_memory(lctx), true);

    const int first_pos = (last > 0 && last < n_tok) ? n_tok - last : 0;
    const int n_pos = n_tok - first_pos;
    const bool want_logits = !out_path.empty();

    FILE *out = nullptr;
    if (want_logits) {
        out = fopen(out_path.c_str(), "wb");
        if (!out) die("cannot open --out");
        uint32_t hdr[8] = {0, 1, (uint32_t)n_tok, (uint32_t)n_vocab, (uint32_t)(topk > 0), (uint32_t)topk,
                           (uint32_t)first_pos, (uint32_t)n_pos};
        memcpy(hdr, "MSLG", 4);
        fwrite(hdr, 4, 8, out);
        fwrite(toks.data(), 4, n_tok, out);
    }

    // ---- prefill in chunks of n_batch, harvesting logits ----
    std::vector<float> chunk_tps;
    double t_pf0 = now_s();
    std::vector<float> last_logits(n_vocab);
    for (int s = 0; s < n_tok; s += nbatch) {
        int n = std::min(nbatch, n_tok - s);
        batch.n_tokens = n;
        for (int j = 0; j < n; j++) {
            int p = s + j;
            batch.token[j] = toks[p];
            batch.pos[j] = p;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (want_logits && p >= first_pos) || p == n_tok - 1;
        }
        double tc = now_s();
        if (llama_decode(lctx, batch) != 0) die("llama_decode failed");
        llama_synchronize(lctx);
        double dt = now_s() - tc;
        chunk_tps.push_back(n / dt);
        fprintf(stderr, "oracle_logits: [%zu] chunk %d..%d  %.2f tok/s  (%.1f s)\n", jn, s, s + n, n / dt, dt);
        for (int j = 0; j < n; j++) {
            int p = s + j;
            if (!batch.logits[j]) continue;
            const float *lg = llama_get_logits_ith(lctx, j);
            if (!lg) die("null logits");
            if (p == n_tok - 1) memcpy(last_logits.data(), lg, sizeof(float) * n_vocab);
            if (!want_logits || p < first_pos) continue;
            if (topk <= 0) {
                fwrite(lg, sizeof(float), n_vocab, out);
            } else {
                float mx = *std::max_element(lg, lg + n_vocab);
                double se = 0; for (int v = 0; v < n_vocab; v++) se += std::exp((double)lg[v] - mx);
                float lse = (float)(mx + std::log(se));
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                                  [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
                float head[2] = {lse, 0.f};
                fwrite(head, 4, 2, out);
                fwrite(idx.data(), 4, topk, out);
                for (int q = 0; q < topk; q++) fwrite(&lg[idx[q]], 4, 1, out);
            }
        }
    }
    double t_prefill = now_s() - t_pf0;
    if (out) fclose(out);
    fprintf(stderr, "oracle_logits: [%zu] prefill %d tokens in %.2f s = %.2f tok/s (load %.1f s)\n", jn, n_tok, t_prefill,
            n_tok / t_prefill, t_load);

    // ---- optional greedy generation ----
    std::string gen_text; std::vector<int> gen_ids; double t_gen = 0;
    if (gen > 0) {
        double tg0 = now_s();
        std::vector<float> lg = last_logits;
        int pos = n_tok;
        for (int g = 0; g < gen; g++) {
            int best = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            gen_ids.push_back(best);
            char buf[256]; int k = llama_token_to_piece(vocab, best, buf, sizeof buf, 0, true);
            if (k > 0) { gen_text.append(buf, k); fwrite(buf, 1, k, stderr); fflush(stderr); }
            if (llama_vocab_is_eog(vocab, best)) break;
            batch.n_tokens = 1;
            batch.token[0] = best; batch.pos[0] = pos++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            if (llama_decode(lctx, batch) != 0) die("decode (gen) failed");
            const float *l = llama_get_logits_ith(lctx, 0);
            memcpy(lg.data(), l, sizeof(float) * n_vocab);
        }
        t_gen = now_s() - tg0;
        fprintf(stderr, "\noracle_logits: generated %zu tokens in %.2f s = %.2f tok/s\n", gen_ids.size(), t_gen,
                gen_ids.size() / t_gen);
    }

    // ---- metadata json ----
    std::string jpath = !out_path.empty() ? out_path + ".json" : !json_path.empty() ? json_path : std::string();
    if (!jpath.empty()) {
        FILE *j = fopen(jpath.c_str(), "w");
        if (j) {
            fprintf(j, "{\n  \"tool\": \"oracle_logits\", \"format_version\": 1,\n");
            fprintf(j, "  \"model\": \"%s\",\n", json_escape(model).c_str());
            fprintf(j, "  \"llama_system_info\": \"%s\",\n", json_escape(llama_print_system_info()).c_str());
            fprintf(j, "  \"n_tokens\": %d, \"n_vocab\": %d, \"mode\": \"%s\", \"topk\": %d, \"first_pos\": %d, \"n_pos\": %d,\n",
                    n_tok, n_vocab, topk > 0 ? "topk" : "full", topk, first_pos, n_pos);
            fprintf(j, "  \"config\": {\"ngl\": %d, \"cpu_moe\": %s, \"mmap\": %s, \"n_ctx\": %u, \"n_batch\": %d, \"n_ubatch\": %d, \"threads\": %d, \"fa\": \"%s\", \"swa_full\": %s},\n",
                    ngl, cpu_moe ? "true" : "false", use_mmap ? "true" : "false", cp.n_ctx, nbatch, nubatch, threads, fa.c_str(),
                    cp.swa_full ? "true" : "false");
            fprintf(j, "  \"job_index\": %zu, \"t_load_s\": %.3f, \"t_prefill_s\": %.3f, \"prefill_tok_s\": %.3f,\n", jn, t_load, t_prefill, n_tok / t_prefill);
            fprintf(j, "  \"chunk_tok_s\": [");
            for (size_t c = 0; c < chunk_tps.size(); c++) fprintf(j, "%s%.3f", c ? ", " : "", chunk_tps[c]);
            fprintf(j, "],\n  \"gen_ids\": [");
            for (size_t c = 0; c < gen_ids.size(); c++) fprintf(j, "%s%d", c ? ", " : "", gen_ids[c]);
            fprintf(j, "],\n  \"gen_text\": \"%s\", \"t_gen_s\": %.3f\n}\n", json_escape(gen_text).c_str(), t_gen);
            fclose(j);
        }
    }
  }
    llama_perf_context_print(lctx);
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
