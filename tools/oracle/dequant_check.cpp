// dequant_check — TEST-ONLY. Dequantizes GGUF quant blocks with stock libggml (Homebrew) so MacSeek's own CPU dequant
// references / Metal kernels can be checked bit-exactly (SPEC §6 gate A). Reads only the requested rows (pread), so it
// does NOT load the model and needs no model lock.
//
//   dequant_check --gguf SPLIT.gguf --tensor NAME [--row0 R] [--rows N] --out OUT.f32 [--raw-out OUT.bin]
//       rows of a 2D/3D tensor (row = ne0 elements; 3D expert tensors: row index runs over ne1*ne2, expert-major)
//       from the split file that contains it (docs/tensors.txt lists the file). Prints type, dims, row bytes.
//   dequant_check --type iq1_s --in BLOCKS.bin --out OUT.f32
//       raw block bytes (whole blocks) -> f32.
//   dequant_check --list-types
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

static void die(const char *m) { fprintf(stderr, "dequant_check: %s\n", m); exit(1); }

static ggml_type type_by_name(const std::string &n) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char *nm = ggml_type_name((ggml_type)t);
        if (nm && n == nm) return (ggml_type)t;
    }
    die("unknown type name (see --list-types)");
    return GGML_TYPE_F32;
}

int main(int argc, char **argv) {
    std::string gguf, tensor, out, raw_out, in, tname;
    long long row0 = 0, rows = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value"); return argv[++i]; };
        if (a == "--gguf") gguf = next();
        else if (a == "--tensor") tensor = next();
        else if (a == "--row0") row0 = atoll(next().c_str());
        else if (a == "--rows") rows = atoll(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--raw-out") raw_out = next();
        else if (a == "--in") in = next();
        else if (a == "--type") tname = next();
        else if (a == "--list-types") {
            for (int t = 0; t < GGML_TYPE_COUNT; t++) {
                const char *nm = ggml_type_name((ggml_type)t);
                if (nm && ggml_get_type_traits((ggml_type)t)->to_float)
                    printf("%2d %-10s blck %lld  bytes/blck %zu\n", t, nm, (long long)ggml_blck_size((ggml_type)t),
                           ggml_row_size((ggml_type)t, ggml_blck_size((ggml_type)t)));
            }
            return 0;
        } else die("bad arg");
    }
    if (out.empty()) die("--out required");
    std::vector<uint8_t> raw;
    ggml_type type;
    long long n_el;
    if (!gguf.empty()) {
        gguf_init_params gp = {/*no_alloc*/ true, /*ctx*/ nullptr};
        ggml_context *meta = nullptr;
        gp.ctx = &meta;
        gguf_context *g = gguf_init_from_file(gguf.c_str(), gp);
        if (!g) die("gguf open failed");
        int64_t id = gguf_find_tensor(g, tensor.c_str());
        if (id < 0) die("tensor not in this split (check docs/tensors.txt for the file)");
        type = gguf_get_tensor_type(g, id);
        ggml_tensor *t = ggml_get_tensor(meta, tensor.c_str());
        long long ne0 = t->ne[0], nrows = ggml_nrows(t);
        size_t rb = ggml_row_size(type, ne0);
        size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        fprintf(stderr, "%s: type %s dims [%lld %lld %lld %lld] row_bytes %zu nrows %lld file_off %zu\n", tensor.c_str(),
                ggml_type_name(type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                rb, nrows, off);
        if (row0 < 0 || row0 + rows > nrows) die("row range out of bounds");
        raw.resize(rb * rows);
        int fd = open(gguf.c_str(), O_RDONLY);
        if (fd < 0) die("open failed");
        if (pread(fd, raw.data(), raw.size(), (off_t)(off + rb * row0)) != (ssize_t)raw.size()) die("short read");
        close(fd);
        n_el = ne0 * rows;
        gguf_free(g);
    } else {
        if (in.empty() || tname.empty()) die("need --gguf/--tensor or --type/--in");
        type = type_by_name(tname);
        FILE *f = fopen(in.c_str(), "rb");
        if (!f) die("cannot open --in");
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        raw.resize(sz);
        if (fread(raw.data(), 1, sz, f) != (size_t)sz) die("read failed");
        fclose(f);
        size_t bb = ggml_row_size(type, ggml_blck_size(type));
        if (raw.size() % bb) die("input is not a whole number of blocks");
        n_el = (long long)(raw.size() / bb) * ggml_blck_size(type);
    }
    const ggml_type_traits *tr = ggml_get_type_traits(type);
    std::vector<float> y(n_el);
    if (type == GGML_TYPE_F32) memcpy(y.data(), raw.data(), n_el * 4);
    else if (!tr->to_float) die("type has no to_float");
    else tr->to_float(raw.data(), y.data(), n_el);
    FILE *f = fopen(out.c_str(), "wb");
    fwrite(y.data(), 4, n_el, f); fclose(f);
    if (!raw_out.empty()) { FILE *r = fopen(raw_out.c_str(), "wb"); fwrite(raw.data(), 1, raw.size(), r); fclose(r); }
    fprintf(stderr, "dequantized %lld elements (%s) -> %s\n", n_el, ggml_type_name(type), out.c_str());
    return 0;
}
