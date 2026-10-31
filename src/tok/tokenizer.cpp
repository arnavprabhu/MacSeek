#include "tok/tokenizer.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_map>

namespace ms::tok {

#include "tok/unicode_table.inc"

namespace {

// ------------------------------------------------------------------------------------------------------------
// Unicode helpers
// ------------------------------------------------------------------------------------------------------------

enum UcClass : uint8_t { UC_OTHER = 0, UC_N = 1, UC_L = 2, UC_P = 3, UC_M = 4, UC_S = 5, UC_WS = 6 };

inline uint8_t uc_class(uint32_t cp) {
  if (cp >= 0x110000) return UC_OTHER;
  return kUcStage2[(size_t)kUcStage1[cp >> 8] * 256 + (cp & 255)];
}

// Decode one codepoint with llama.cpp's acceptance rules (overlongs rejected, surrogates accepted, <= 0x10FFFF).
// Returns the sequence length (1..4) or 0 if invalid at s[i].
inline int utf8_decode_lenient(const unsigned char* s, size_t n, size_t i, uint32_t& cp) {
  unsigned char c = s[i];
  if (c < 0x80) { cp = c; return 1; }
  if (!(c & 0x40)) return 0;
  if (!(c & 0x20)) {
    if (i + 1 >= n || (s[i + 1] & 0xC0) != 0x80) return 0;
    cp = ((uint32_t)(c & 0x1F) << 6) | (s[i + 1] & 0x3F);
    return cp < 0x80 ? 0 : 2;
  }
  if (!(c & 0x10)) {
    if (i + 2 >= n || (s[i + 1] & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80) return 0;
    cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
    return cp < 0x800 ? 0 : 3;
  }
  if (!(c & 0x08)) {
    if (i + 3 >= n || (s[i + 1] & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80 || (s[i + 3] & 0xC0) != 0x80) return 0;
    cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) | ((uint32_t)(s[i + 2] & 0x3F) << 6) |
         (s[i + 3] & 0x3F);
    return (cp < 0x10000 || cp > 0x10FFFF) ? 0 : 4;
  }
  return 0;
}

inline void utf8_put(std::string& s, uint32_t cp) {
  if (cp < 0x80) s += (char)cp;
  else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
  else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
  else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 63)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
}

// GPT-2 bytes_to_unicode().
struct ByteMap {
  uint32_t byte_to_cp[256];
  int16_t cp_to_byte[512];
  ByteMap() {
    for (auto& v : cp_to_byte) v = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
      bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      byte_to_cp[b] = direct ? (uint32_t)b : (uint32_t)(256 + n++);
      cp_to_byte[byte_to_cp[b]] = (int16_t)b;
    }
  }
};
const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

// ------------------------------------------------------------------------------------------------------------
// Pre-tokenizer predicate tables over the "collapsed" byte of each codepoint (llama.cpp convention):
// ASCII codepoints are themselves; others: White_Space -> 0x0B, N -> 0xD1, L -> 0xD2, P -> 0xD3, M -> 0xD4,
// S -> 0xD5, anything else -> 0xD0.
// ------------------------------------------------------------------------------------------------------------

constexpr uint8_t kCollapse[7] = {0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0x0B};

struct Preds {
  bool N[256], L[256], LM[256], PS[256], X[256], Space[256], AsciiPunct[256], Alpha[256], CRLF[256];
  Preds() {
    memset(this, 0, sizeof *this);
    for (int b = 0; b < 256; b++) {
      bool asciiAlpha = (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z');
      bool asciiPunct = (b >= 0x21 && b <= 0x2F) || (b >= 0x3A && b <= 0x40) || (b >= 0x5B && b <= 0x60) ||
                        (b >= 0x7B && b <= 0x7E);
      Alpha[b] = asciiAlpha;
      AsciiPunct[b] = asciiPunct;
      N[b] = (b >= '0' && b <= '9') || b == 0xD1;
      L[b] = asciiAlpha || b == 0xD2;
      LM[b] = L[b] || b == 0xD4;
      PS[b] = asciiPunct || b == 0xD3 || b == 0xD5;
      Space[b] = b == ' ' || (b >= 0x09 && b <= 0x0D);  // std::regex \s in the "C" locale
      CRLF[b] = b == '\r' || b == '\n';
      X[b] = !(b == '\r' || b == '\n' || L[b] || PS[b]);  // [^\r\n\p{L}\p{P}\p{S}]
    }
  }
};
const Preds& preds() {
  static const Preds p;
  return p;
}

inline bool is_cjk_kana(uint32_t cp) {  // [一-龥぀-ゟ゠-ヿ]
  return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
}

// ------------------------------------------------------------------------------------------------------------
// Merge table: open addressing (left_id, right_id) -> (rank, merged_id)
// ------------------------------------------------------------------------------------------------------------
struct MergeTable {
  struct Slot {
    uint64_t key;
    int32_t rank;
    int32_t id;
  };
  std::vector<Slot> slots;
  uint64_t mask = 0;
  static constexpr uint64_t kEmpty = ~0ull;
  static inline uint64_t hash(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    return k;
  }
  void init(size_t n) {
    size_t cap = 1;
    while (cap < n * 2) cap <<= 1;
    slots.assign(cap, Slot{kEmpty, 0, 0});
    mask = cap - 1;
  }
  void insert(int32_t a, int32_t b, int32_t rank, int32_t id) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (uint64_t h = hash(k) & mask;; h = (h + 1) & mask) {
      if (slots[h].key == kEmpty) { slots[h] = Slot{k, rank, id}; return; }
      if (slots[h].key == k) return;  // keep the first (lowest) rank
    }
  }
  inline const Slot* find(int32_t a, int32_t b) const {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (uint64_t h = hash(k) & mask;; h = (h + 1) & mask) {
      const Slot& s = slots[h];
      if (s.key == k) return &s;
      if (s.key == kEmpty) return nullptr;
    }
  }
};

struct TrieNode {
  std::vector<std::pair<uint8_t, int32_t>> next;  // sorted by byte
  int32_t token = -1;
};

struct StrHash {
  size_t operator()(std::string_view s) const noexcept {
    // FNV-1a 64 over up to the whole piece (pieces are short)
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return (size_t)h;
  }
};

}  // namespace

// ------------------------------------------------------------------------------------------------------------

struct Tokenizer::Impl {
  VocabData v;
  std::vector<std::string> pieces;  // decoded bytes per token
  std::unordered_map<std::string_view, int32_t, StrHash> text_to_id;  // views into v.tokens
  int32_t byte_tok[256];
  MergeTable merges;
  // special-token partition (llama.cpp order: text length descending, std::sort on ids ascending)
  std::vector<int32_t> specials;
  std::vector<int32_t> special_rank;  // per token id, -1 if not special
  std::vector<TrieNode> trie;
  bool trie_first[256] = {};

  // ---- BPE ----
  void bpe(const unsigned char* s, size_t n, std::vector<int32_t>& out) const {
    if (n == 1) { out.push_back(byte_tok[s[0]]); return; }
    if (n <= 96) { bpe_small(s, n, out); return; }
    bpe_large(s, n, out);
  }

  // O(n^2) for short pieces: repeatedly merge the lowest-rank adjacent pair (leftmost on ties).
  void bpe_small(const unsigned char* s, size_t n, std::vector<int32_t>& out) const {
    int32_t ids[96];
    int32_t rk[96];  // rank of pair (i, i+1), INT32_MAX if none
    int32_t mid[96];
    size_t m = n;
    for (size_t i = 0; i < n; i++) ids[i] = byte_tok[s[i]];
    auto pair = [&](size_t i) {
      const MergeTable::Slot* sl = merges.find(ids[i], ids[i + 1]);
      if (sl) { rk[i] = sl->rank; mid[i] = sl->id; } else { rk[i] = INT32_MAX; }
    };
    for (size_t i = 0; i + 1 < m; i++) pair(i);
    while (m > 1) {
      size_t best = 0;
      int32_t br = rk[0];
      for (size_t i = 1; i + 1 < m; i++)
        if (rk[i] < br) { br = rk[i]; best = i; }
      if (br == INT32_MAX) break;
      ids[best] = mid[best];
      // remove ids[best+1]
      for (size_t i = best + 1; i + 1 < m; i++) { ids[i] = ids[i + 1]; rk[i] = rk[i + 1]; mid[i] = mid[i + 1]; }
      m--;
      if (best > 0) pair(best - 1);
      if (best + 1 < m) pair(best);
    }
    out.insert(out.end(), ids, ids + m);
  }

  // Heap + linked list (same semantics as llama.cpp's llm_bigram_bpe queue) for long pieces.
  void bpe_large(const unsigned char* s, size_t n, std::vector<int32_t>& out) const {
    struct Sym { int32_t id; int32_t prev, next; bool alive; };
    std::vector<Sym> sy(n);
    for (size_t i = 0; i < n; i++) sy[i] = Sym{byte_tok[s[i]], (int32_t)i - 1, i + 1 < n ? (int32_t)i + 1 : -1, true};
    struct Big { int32_t rank; int32_t left, right; int32_t lid, rid; };
    auto cmp = [](const Big& a, const Big& b) { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); };
    std::priority_queue<Big, std::vector<Big>, decltype(cmp)> q(cmp);
    auto add = [&](int32_t l, int32_t r) {
      if (l < 0 || r < 0) return;
      const MergeTable::Slot* sl = merges.find(sy[l].id, sy[r].id);
      if (sl) q.push(Big{sl->rank, l, r, sy[l].id, sy[r].id});
    };
    for (size_t i = 0; i + 1 < n; i++) add((int32_t)i, (int32_t)i + 1);
    while (!q.empty()) {
      Big b = q.top();
      q.pop();
      Sym& L = sy[b.left];
      Sym& R = sy[b.right];
      if (!L.alive || !R.alive || L.next != b.right || L.id != b.lid || R.id != b.rid) continue;  // stale
      const MergeTable::Slot* sl = merges.find(L.id, R.id);
      L.id = sl->id;
      R.alive = false;
      L.next = R.next;
      if (R.next >= 0) sy[R.next].prev = b.left;
      add(L.prev, b.left);
      add(b.left, L.next);
    }
    for (int32_t i = 0; i >= 0; i = sy[i].next) out.push_back(sy[i].id);
  }

  // ---- pre-tokenizer + BPE over one raw fragment ----
  struct Scratch {
    std::vector<uint32_t> off;  // byte offset of codepoint i (size n+1)
    std::vector<uint8_t> cls;   // collapsed byte per codepoint
    std::vector<uint32_t> cps;
    std::string sanitized;
    std::unordered_map<std::string_view, std::pair<uint32_t, uint32_t>, StrHash> cache;  // piece -> (start,len) in ids_store
    std::vector<int32_t> ids_store;
  };

  template <class Emit>
  void pretok(std::string_view frag, Scratch& sc, Emit&& emit) const {
    const unsigned char* s = (const unsigned char*)frag.data();
    size_t n = frag.size();
    sc.off.clear();
    sc.cls.clear();
    sc.cps.clear();
    sc.off.reserve(n + 1);
    sc.cls.reserve(n);
    sc.cps.reserve(n);
    for (size_t i = 0; i < n;) {
      uint32_t cp;
      int len = s[i] < 0x80 ? (cp = s[i], 1) : utf8_decode_lenient(s, n, i, cp);
      if (len == 0) { len = 1; cp = 0xFFFD; }  // only reached on sanitized input (never)
      sc.off.push_back((uint32_t)i);
      sc.cps.push_back(cp);
      sc.cls.push_back(cp < 128 ? (uint8_t)cp : kCollapse[uc_class(cp)]);
      i += (size_t)len;
    }
    sc.off.push_back((uint32_t)n);
    const size_t m = sc.cls.size();
    const uint8_t* c = sc.cls.data();
    const uint32_t* cps = sc.cps.data();
    const Preds& P = preds();

    auto emit_piece = [&](size_t a, size_t b) { emit(frag.substr(sc.off[a], sc.off[b] - sc.off[a])); };

    // regex 3: [punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ |
    //          \s+(?!\S) | \s+
    auto match3 = [&](size_t i, size_t e) -> size_t {
      const uint8_t b = c[i];
      const bool has1 = i + 1 < e;
      if (P.AsciiPunct[b] && has1 && P.Alpha[c[i + 1]]) {
        size_t j = i + 2;
        while (j < e && P.Alpha[c[j]]) j++;
        return j;
      }
      if (P.X[b] && has1 && P.LM[c[i + 1]]) {
        size_t j = i + 2;
        while (j < e && P.LM[c[j]]) j++;
        return j;
      }
      if (P.LM[b]) {
        size_t j = i + 1;
        while (j < e && P.LM[c[j]]) j++;
        return j;
      }
      if ((b == ' ' && has1 && P.PS[c[i + 1]]) || P.PS[b]) {
        size_t j = (b == ' ') ? i + 2 : i + 1;
        while (j < e && P.PS[c[j]]) j++;
        while (j < e && P.CRLF[c[j]]) j++;
        return j;
      }
      if (P.Space[b]) {
        size_t k = i;
        size_t last_nl = SIZE_MAX;
        while (k < e && P.Space[c[k]]) {
          if (P.CRLF[c[k]]) last_nl = k;
          k++;
        }
        if (last_nl != SIZE_MAX) return last_nl + 1;  // \s*[\r\n]+
        if (k == e) return k;                         // \s+(?!\S) at chunk end
        if (k - 1 > i) return k - 1;                  // \s+(?!\S) leaving one space
        return k;                                     // \s+
      }
      return i;
    };
    auto pass3 = [&](size_t s0, size_t e) {
      size_t i = s0, gap = s0;
      while (i < e) {
        size_t j = match3(i, e);
        if (j > i) {
          if (gap < i) emit_piece(gap, i);
          emit_piece(i, j);
          i = gap = j;
        } else {
          i++;
        }
      }
      if (gap < e) emit_piece(gap, e);
    };
    // regex 2: [一-龥぀-ゟ゠-ヿ]+
    auto pass2 = [&](size_t s0, size_t e) {
      size_t i = s0, gap = s0;
      while (i < e) {
        if (is_cjk_kana(cps[i])) {
          size_t j = i + 1;
          while (j < e && is_cjk_kana(cps[j])) j++;
          if (gap < i) pass3(gap, i);
          pass3(i, j);
          i = gap = j;
        } else {
          i++;
        }
      }
      if (gap < e) pass3(gap, e);
    };
    // regex 1: \p{N}{1,3}
    size_t i = 0, gap = 0;
    while (i < m) {
      if (P.N[c[i]]) {
        size_t j = i + 1;
        while (j < m && j < i + 3 && P.N[c[j]]) j++;
        if (gap < i) pass2(gap, i);
        pass2(i, j);
        i = gap = j;
      } else {
        i++;
      }
    }
    if (gap < m) pass2(gap, m);
  }

  // Returns a view of frag where invalid UTF-8 bytes are replaced by U+FFFD (llama.cpp behaviour).
  std::string_view sanitize(std::string_view frag, Scratch& sc) const {
    const unsigned char* s = (const unsigned char*)frag.data();
    size_t n = frag.size(), i = 0;
    for (; i < n;) {
      if (s[i] < 0x80) { i++; continue; }
      uint32_t cp;
      int len = utf8_decode_lenient(s, n, i, cp);
      if (!len) break;
      i += (size_t)len;
    }
    if (i == n) return frag;
    sc.sanitized.assign(frag.data(), i);
    while (i < n) {
      uint32_t cp;
      int len = s[i] < 0x80 ? (cp = s[i], 1) : utf8_decode_lenient(s, n, i, cp);
      if (!len) { sc.sanitized += "\xEF\xBF\xBD"; i++; continue; }
      sc.sanitized.append(frag.data() + i, (size_t)len);
      i += (size_t)len;
    }
    return sc.sanitized;
  }

  void encode_raw(std::string_view frag, std::vector<int32_t>& out, Scratch& sc) const {
    if (frag.empty()) return;
    std::string_view text = sanitize(frag, sc);
    // NB: if sanitized, `text` points into sc.sanitized; cached views must not outlive it -> clear cache then.
    const bool tmp = text.data() != frag.data();
    pretok(text, sc, [&](std::string_view piece) {
      if (piece.size() <= 48 && !tmp) {
        auto it = sc.cache.find(piece);
        if (it != sc.cache.end()) {
          out.insert(out.end(), sc.ids_store.begin() + it->second.first,
                     sc.ids_store.begin() + it->second.first + it->second.second);
          return;
        }
        size_t before = out.size();
        bpe((const unsigned char*)piece.data(), piece.size(), out);
        uint32_t start = (uint32_t)sc.ids_store.size();
        sc.ids_store.insert(sc.ids_store.end(), out.begin() + (long)before, out.end());
        sc.cache.emplace(piece, std::make_pair(start, (uint32_t)(out.size() - before)));
        return;
      }
      bpe((const unsigned char*)piece.data(), piece.size(), out);
    });
  }

  // ---- special-token partition ----
  struct Frag {
    size_t off, len;
    int32_t id;  // -1 = raw text
  };
  void partition(std::string_view text, bool parse_special, std::vector<Frag>& frags) const {
    struct Occ { int32_t rank; uint32_t pos; uint32_t len; int32_t id; };
    std::vector<Occ> occ;
    const unsigned char* s = (const unsigned char*)text.data();
    const size_t n = text.size();
    for (size_t p = 0; p < n; p++) {
      if (!trie_first[s[p]]) continue;
      int32_t node = 0;
      for (size_t q = p; q < n; q++) {
        const auto& nx = trie[(size_t)node].next;
        auto it = std::lower_bound(nx.begin(), nx.end(), std::make_pair(s[q], (int32_t)-1));
        if (it == nx.end() || it->first != s[q]) break;
        node = it->second;
        int32_t tid = trie[(size_t)node].token;
        if (tid >= 0) {
          TokenType t = (TokenType)v.types[(size_t)tid];
          if (parse_special || !(t == kTokControl || t == kTokUnknown))
            occ.push_back(Occ{special_rank[(size_t)tid], (uint32_t)p, (uint32_t)(q + 1 - p), tid});
        }
      }
    }
    frags.clear();
    if (occ.empty()) { frags.push_back(Frag{0, n, -1}); return; }
    std::sort(occ.begin(), occ.end(), [](const Occ& a, const Occ& b) {
      return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos);
    });
    // accept greedily in priority order if not overlapping an accepted interval
    std::vector<Occ> acc;
    std::vector<uint8_t> covered;
    covered.assign(n, 0);
    for (const Occ& o : occ) {
      bool free = true;
      for (uint32_t k = 0; k < o.len; k++)
        if (covered[o.pos + k]) { free = false; break; }
      if (!free) continue;
      memset(&covered[o.pos], 1, o.len);
      acc.push_back(o);
    }
    std::sort(acc.begin(), acc.end(), [](const Occ& a, const Occ& b) { return a.pos < b.pos; });
    size_t cur = 0;
    for (const Occ& o : acc) {
      if (o.pos > cur) frags.push_back(Frag{cur, o.pos - cur, -1});
      frags.push_back(Frag{o.pos, o.len, o.id});
      cur = o.pos + o.len;
    }
    if (cur < n) frags.push_back(Frag{cur, n - cur, -1});
  }
};

// ------------------------------------------------------------------------------------------------------------
// GGUF metadata reader (private; tokenizer keys only)
// ------------------------------------------------------------------------------------------------------------

bool read_gguf_vocab(const std::string& path, VocabData& out, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = path + ": " + m;
    return false;
  };
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return fail("cannot open");
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); return fail("stat failed"); }
  size_t fsize = (size_t)st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return fail("mmap failed");
  const uint8_t* b = (const uint8_t*)map;
  const uint8_t* p = b;
  const uint8_t* e = b + fsize;
  bool ok = true;
  auto need = [&](size_t k) { if ((size_t)(e - p) < k) { ok = false; return false; } return true; };
  auto rd_u32 = [&]() -> uint32_t { uint32_t v = 0; if (need(4)) { memcpy(&v, p, 4); p += 4; } return v; };
  auto rd_u64 = [&]() -> uint64_t { uint64_t v = 0; if (need(8)) { memcpy(&v, p, 8); p += 8; } return v; };
  auto rd_str = [&]() -> std::string_view {
    uint64_t n = rd_u64();
    if (!ok || !need(n)) return {};
    std::string_view s((const char*)p, n);
    p += n;
    return s;
  };
  static const size_t kScalar[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  auto rd_int = [&](uint32_t t) -> int64_t {
    int64_t v = 0;
    switch (t) {
      case 0: if (need(1)) v = *p; p += 1; break;
      case 1: if (need(1)) v = (int8_t)*p; p += 1; break;
      case 2: { uint16_t x = 0; if (need(2)) memcpy(&x, p, 2); p += 2; v = x; break; }
      case 3: { int16_t x = 0; if (need(2)) memcpy(&x, p, 2); p += 2; v = x; break; }
      case 4: v = rd_u32(); break;
      case 5: v = (int32_t)rd_u32(); break;
      case 7: if (need(1)) v = *p; p += 1; break;
      case 10: v = (int64_t)rd_u64(); break;
      case 11: v = (int64_t)rd_u64(); break;
      default: ok = false;
    }
    return v;
  };
  if (!need(4) || memcmp(p, "GGUF", 4) != 0) { munmap(map, fsize); return fail("not a GGUF file"); }
  p += 4;
  uint32_t ver = rd_u32();
  if (ver < 2) { munmap(map, fsize); return fail("unsupported GGUF version"); }
  rd_u64();  // tensor count
  uint64_t nkv = rd_u64();
  out = VocabData();
  for (uint64_t k = 0; k < nkv && ok; k++) {
    std::string_view key = rd_str();
    uint32_t t = rd_u32();
    if (!ok) break;
    if (t == 8) {
      std::string_view s = rd_str();
      if (key == "tokenizer.ggml.model") out.model = s;
      else if (key == "tokenizer.ggml.pre") out.pre = s;
      else if (key == "tokenizer.chat_template") out.chat_template = s;
    } else if (t == 9) {
      uint32_t et = rd_u32();
      uint64_t cnt = rd_u64();
      if (!ok) break;
      if (et == 8) {
        std::vector<std::string>* dst = nullptr;
        if (key == "tokenizer.ggml.tokens") dst = &out.tokens;
        else if (key == "tokenizer.ggml.merges") dst = &out.merges;
        if (dst) dst->reserve(cnt);
        for (uint64_t i = 0; i < cnt && ok; i++) {
          std::string_view s = rd_str();
          if (dst) dst->emplace_back(s);
        }
      } else if (et < 13 && kScalar[et]) {
        if (!need(kScalar[et] * cnt)) break;
        if (key == "tokenizer.ggml.token_type") {
          out.types.resize(cnt);
          for (uint64_t i = 0; i < cnt; i++) out.types[i] = (int32_t)rd_int(et);
        } else {
          p += kScalar[et] * cnt;
        }
      } else {
        ok = false;  // nested arrays: not present in tokenizer metadata
      }
    } else if (t < 13 && kScalar[t]) {
      if (t == 6 || t == 12) { p += kScalar[t]; continue; }
      int64_t v = rd_int(t);
      if (key == "tokenizer.ggml.bos_token_id") out.bos = (int32_t)v;
      else if (key == "tokenizer.ggml.eos_token_id") out.eos = (int32_t)v;
      else if (key == "tokenizer.ggml.padding_token_id") out.pad = (int32_t)v;
      else if (key == "tokenizer.ggml.add_bos_token") out.add_bos = v != 0;
      else if (key == "tokenizer.ggml.add_eos_token") out.add_eos = v != 0;
    } else {
      ok = false;
    }
  }
  munmap(map, fsize);
  if (!ok) return fail("truncated or malformed metadata");
  if (out.tokens.empty()) return fail("no tokenizer.ggml.tokens");
  return true;
}

// ------------------------------------------------------------------------------------------------------------

Tokenizer::Tokenizer() : d_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::load_from_gguf(const std::string& path, std::string* err) {
  VocabData vd;
  if (!read_gguf_vocab(path, vd, err)) return nullptr;
  auto t = std::make_unique<Tokenizer>();
  if (!t->init(std::move(vd), err)) return nullptr;
  return t;
}

bool Tokenizer::init(VocabData data, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    return false;
  };
  Impl& d = *d_;
  d.v = std::move(data);
  if (d.v.model != "gpt2") return fail("unsupported tokenizer model '" + d.v.model + "' (need gpt2)");
  if (d.v.pre != "joyai-llm" && d.v.pre != "deepseek-v3" && d.v.pre != "hunyuan-dense")
    return fail("unsupported pre-tokenizer '" + d.v.pre + "'");
  const size_t n = d.v.tokens.size();
  if (d.v.types.size() != n) d.v.types.assign(n, kTokNormal);
  d.text_to_id.clear();
  d.text_to_id.reserve(n * 2);
  for (size_t i = 0; i < n; i++) d.text_to_id.emplace(d.v.tokens[i], (int32_t)i);  // first wins

  // decoded pieces
  const ByteMap& bm = byte_map();
  d.pieces.assign(n, std::string());
  for (size_t i = 0; i < n; i++) {
    const std::string& t = d.v.tokens[i];
    TokenType ty = (TokenType)d.v.types[i];
    if (ty == kTokControl || ty == kTokUserDefined || ty == kTokUnknown) { d.pieces[i] = t; continue; }
    if (ty == kTokUnused) continue;
    std::string out;
    const unsigned char* s = (const unsigned char*)t.data();
    for (size_t k = 0; k < t.size();) {
      uint32_t cp;
      int len = utf8_decode_lenient(s, t.size(), k, cp);
      if (!len) { out += (char)s[k]; k++; continue; }
      if (cp < 512 && bm.cp_to_byte[cp] >= 0) out += (char)bm.cp_to_byte[cp];
      else out.append(t, k, (size_t)len);  // not byte-level: keep as is
      k += (size_t)len;
    }
    d.pieces[i] = std::move(out);
  }

  // byte tokens
  for (int b = 0; b < 256; b++) {
    std::string s;
    utf8_put(s, bm.byte_to_cp[b]);
    auto it = d.text_to_id.find(s);
    if (it == d.text_to_id.end()) return fail("vocab lacks byte token for byte " + std::to_string(b));
    d.byte_tok[b] = it->second;
  }

  // merges -> id table
  d.merges.init(d.v.merges.size());
  size_t bad = 0;
  for (size_t r = 0; r < d.v.merges.size(); r++) {
    const std::string& m = d.v.merges[r];
    size_t sp = m.find(' ', 1);
    if (sp == std::string::npos) { bad++; continue; }
    std::string_view a(m.data(), sp), bstr(m.data() + sp + 1, m.size() - sp - 1);
    auto ia = d.text_to_id.find(a), ib = d.text_to_id.find(bstr);
    std::string ab = std::string(a) + std::string(bstr);
    auto iab = d.text_to_id.find(ab);
    if (ia == d.text_to_id.end() || ib == d.text_to_id.end() || iab == d.text_to_id.end()) { bad++; continue; }
    d.merges.insert(ia->second, ib->second, (int32_t)r, iab->second);
  }
  if (bad) return fail(std::to_string(bad) + " merges reference tokens missing from the vocab (id-BPE not exact)");

  // special tokens (llama.cpp: CONTROL | USER_DEFINED | UNKNOWN, sorted by text length desc with std::sort)
  d.specials.clear();
  for (size_t i = 0; i < n; i++) {
    TokenType ty = (TokenType)d.v.types[i];
    if (ty == kTokControl || ty == kTokUserDefined || ty == kTokUnknown) d.specials.push_back((int32_t)i);
  }
  const auto& toks = d.v.tokens;
  std::sort(d.specials.begin(), d.specials.end(),
            [&](int32_t a, int32_t b) { return toks[(size_t)a].size() > toks[(size_t)b].size(); });
  d.special_rank.assign(n, -1);
  for (size_t r = 0; r < d.specials.size(); r++) d.special_rank[(size_t)d.specials[r]] = (int32_t)r;
  d.trie.assign(1, TrieNode());
  memset(d.trie_first, 0, sizeof d.trie_first);
  for (int32_t id : d.specials) {
    const std::string& t = toks[(size_t)id];
    if (t.empty()) continue;
    d.trie_first[(unsigned char)t[0]] = true;
    int32_t node = 0;
    for (unsigned char c : t) {
      auto& nx = d.trie[(size_t)node].next;
      auto it = std::lower_bound(nx.begin(), nx.end(), std::make_pair(c, (int32_t)-1));
      if (it != nx.end() && it->first == c) { node = it->second; continue; }
      int32_t nn = (int32_t)d.trie.size();
      nx.insert(it, std::make_pair(c, nn));
      d.trie.emplace_back();
      node = nn;
    }
    if (d.trie[(size_t)node].token < 0) d.trie[(size_t)node].token = id;
  }
  return true;
}

void Tokenizer::encode_append(std::string_view text, std::vector<int32_t>& out, bool parse_special) const {
  if (text.empty()) return;
  std::vector<Impl::Frag> frags;
  d_->partition(text, parse_special, frags);
  Impl::Scratch sc;
  sc.cache.reserve(4096);
  out.reserve(out.size() + text.size() / 3);
  for (const auto& f : frags) {
    if (f.id >= 0) out.push_back(f.id);
    else d_->encode_raw(text.substr(f.off, f.len), out, sc);
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special, bool add_bos) const {
  std::vector<int32_t> out;
  if (add_bos && d_->v.bos >= 0) out.push_back(d_->v.bos);
  encode_append(text, out, parse_special);
  return out;
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view text) const {
  std::vector<std::string> out;
  Impl::Scratch sc;
  std::string_view t = d_->sanitize(text, sc);
  std::string copy(t);
  d_->pretok(copy, sc, [&](std::string_view p) { out.emplace_back(p); });
  return out;
}

std::string_view Tokenizer::piece(int32_t id, bool special) const {
  if (id < 0 || (size_t)id >= d_->pieces.size()) return {};
  if (!special && d_->v.types[(size_t)id] == kTokControl) return {};
  return d_->pieces[(size_t)id];
}

std::string Tokenizer::decode(const int32_t* ids, size_t n, bool special) const {
  std::string s;
  for (size_t i = 0; i < n; i++) s += piece(ids[i], special);
  return s;
}

int32_t Tokenizer::find_token(std::string_view text) const {
  auto it = d_->text_to_id.find(text);
  return it == d_->text_to_id.end() ? -1 : it->second;
}
const std::string& Tokenizer::token_text(int32_t id) const {
  static const std::string empty;
  if (id < 0 || (size_t)id >= d_->v.tokens.size()) return empty;
  return d_->v.tokens[(size_t)id];
}
TokenType Tokenizer::type(int32_t id) const {
  if (id < 0 || (size_t)id >= d_->v.types.size()) return kTokUndefined;
  return (TokenType)d_->v.types[(size_t)id];
}
bool Tokenizer::is_eog(int32_t id) const { return id == d_->v.eos; }
int32_t Tokenizer::n_vocab() const { return (int32_t)d_->v.tokens.size(); }
int32_t Tokenizer::bos() const { return d_->v.bos; }
int32_t Tokenizer::eos() const { return d_->v.eos; }
bool Tokenizer::add_bos() const { return d_->v.add_bos; }
const std::string& Tokenizer::chat_template() const { return d_->v.chat_template; }
const std::string& Tokenizer::pre() const { return d_->v.pre; }

// ------------------------------------------------------------------------------------------------------------
// UTF-8 streaming
// ------------------------------------------------------------------------------------------------------------

// Strict decoder: returns length of a valid sequence at s[i], 0 if invalid, -1 if truncated (valid so far).
static int utf8_strict(const unsigned char* s, size_t n, size_t i) {
  unsigned char c = s[i];
  if (c < 0x80) return 1;
  int len;
  uint32_t lo = 0x80, hi = 0xBF;  // allowed range of the 2nd byte
  if (c >= 0xC2 && c <= 0xDF) len = 2;
  else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
  else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
  else return 0;
  for (int k = 1; k < len; k++) {
    if (i + (size_t)k >= n) return -1;
    unsigned char x = s[i + (size_t)k];
    if (k == 1 ? (x < lo || x > hi) : ((x & 0xC0) != 0x80)) return 0;
  }
  return len;
}

size_t utf8_complete_prefix(std::string_view sv) {
  const unsigned char* s = (const unsigned char*)sv.data();
  size_t n = sv.size(), i = 0;
  while (i < n) {
    int l = utf8_strict(s, n, i);
    if (l < 0) return i;
    i += l == 0 ? 1 : (size_t)l;
  }
  return n;
}

std::string utf8_sanitize(std::string_view sv) {
  const unsigned char* s = (const unsigned char*)sv.data();
  size_t n = sv.size(), i = 0;
  std::string out;
  out.reserve(n);
  while (i < n) {
    int l = utf8_strict(s, n, i);
    if (l <= 0) { out += "\xEF\xBF\xBD"; i++; continue; }
    out.append(sv.data() + i, (size_t)l);
    i += (size_t)l;
  }
  return out;
}

std::string StreamDecoder::push(int32_t id) { return push_bytes(t_.piece(id, special_)); }

std::string StreamDecoder::push_bytes(std::string_view bytes) {
  pending_.append(bytes);
  size_t k = utf8_complete_prefix(pending_);
  std::string out = utf8_sanitize(std::string_view(pending_).substr(0, k));
  pending_.erase(0, k);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out = utf8_sanitize(pending_);
  pending_.clear();
  return out;
}

std::string default_gguf_path() {
  const char* e = getenv("MACSEEK_GGUF");
  if (e && *e) return e;
  const char* home = getenv("HOME");
  return std::string(home ? home : "/Users/arnav") +
         "/Unsloth Models/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/"
         "8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf";
}

}  // namespace ms::tok
