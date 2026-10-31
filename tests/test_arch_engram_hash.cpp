// Engram n-gram hashing (CPU): streaming hasher vs an independent whole-sequence implementation (synthetic params,
// chunk splits, dead tokens, state save/restore), plus real GGUF metadata checks and golden values produced by
// tools/engram_hash_check.py (a literal port of engram.py). `--dump tok,tok,... split` prints rows for the script.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "model/engram_hash.hpp"
#include "ref/arch_testutil.hpp"

using namespace ms;
using P = EngramHashParams;

static const uint32_t kGolden[6][48] = {
    {5702652, 24393184, 39183301, 61503783, 78984481, 90144770, 103724983, 121476532, 131476717, 152385926, 164143652, 191599181, 198145581, 217419202, 238329687, 247482304, 257520185, 283540304, 288689443, 315264704, 322314652, 348890037, 367865216, 380066193, 398368132, 413983977, 416722612, 445215939, 451462373, 468281219, 492500607, 504610715, 516231352, 534276620, 558812120, 562901422, 577214091, 595616364, 610286750, 636511414, 645368429, 663868965, 685370227, 696871016, 712373204, 733378787, 747879910, 756382139},
    {11844763, 28836334, 43475432, 48608257, 64376231, 90984045, 104291392, 118825166, 129595260, 152154682, 164285414, 179795391, 204772272, 209291606, 232109686, 242778020, 261161045, 273478120, 295330354, 306264295, 329556888, 340499523, 361922719, 375216216, 390100262, 403511473, 426021825, 446372354, 448550401, 474568600, 494544361, 496593389, 519124956, 535930051, 545622475, 563245628, 584940811, 592298563, 620652932, 624796002, 649768732, 662257331, 679110642, 688626952, 712892058, 722184303, 736039435, 757320098},
    {5273542, 29042104, 32665903, 61707111, 73438097, 88060333, 111380052, 123350512, 134446345, 151809867, 175239764, 189591453, 202643327, 209245464, 227244070, 253766713, 266318840, 272239103, 303760260, 311539758, 332733569, 340534301, 358233949, 379313373, 391272078, 400160470, 423681206, 434061093, 463261669, 468697443, 492708636, 496913728, 519180189, 533911347, 550806833, 564312900, 587634903, 592594765, 622491517, 632539331, 640494416, 661947806, 678725602, 701941499, 712522224, 724445546, 742902281, 755206161},
    {10515117, 16601983, 40390000, 63191501, 69720805, 90117947, 101732886, 126588211, 128492828, 158628385, 171476505, 187367144, 199740995, 210670306, 234992256, 254679077, 264979992, 278010953, 298561880, 304022733, 330177015, 339177404, 360161703, 371744248, 386828832, 403749642, 430947906, 437257327, 449665611, 476830422, 486653894, 500944445, 516099311, 543238011, 549199194, 561120651, 585431761, 606601028, 620853900, 630726831, 652675162, 659710658, 678670737, 694504796, 715328375, 731249595, 748343639, 756946084},
    {11457147, 23966341, 47011294, 55906811, 79039383, 92560750, 107410962, 122069349, 141865246, 147385191, 168336610, 176032975, 194900810, 213164282, 225595965, 248569867, 269824320, 287875667, 292010534, 318289079, 335886085, 346398149, 357565496, 383320643, 388739563, 400910350, 417553233, 437156556, 455910843, 467779732, 481699820, 507945082, 522451085, 543658992, 554478308, 569162652, 577537245, 597404987, 621648356, 637636549, 647363123, 669254994, 686279004, 689097777, 711651924, 725971690, 747831350, 763504626},
    {5412672, 18319159, 35492627, 60880252, 74962984, 95657733, 98005115, 127351390, 139026703, 155539529, 167539037, 176289752, 205513951, 218109004, 236480277, 255195893, 270585406, 272787863, 293904384, 313972579, 333244006, 338989282, 366747986, 371986804, 390945986, 411270885, 429261001, 435921168, 463252092, 470874638, 492855228, 503847071, 515688836, 543447129, 547964844, 574052828, 588358775, 604469374, 611136518, 628790282, 649802603, 662721504, 674620227, 688025319, 712609507, 734397181, 741858387, 764611678},
};

static const char* kKeys[] = {"deepseek41.engram.token_map", "deepseek41.engram.pad_token_id",
                              "deepseek41.ple.layer_multipliers", "deepseek41.ple.head_vocab_sizes",
                              "deepseek41.ple.head_offsets"};

static bool load_real(EngramHashParams& p) {
  std::map<std::string, std::vector<int64_t>> m;
  if (!arch_test::read_meta_int_arrays(std::vector<std::string>(kKeys, kKeys + 5), m)) return false;
  return p.init(m[kKeys[0]], m[kKeys[1]][0], m[kKeys[2]], m[kKeys[3]], m[kKeys[4]]);
}

// Independent reference over a whole sequence (engram.py semantics with absolute positions).
static std::vector<uint32_t> whole(const P& p, const std::vector<int32_t>& toks, const std::vector<bool>* mask) {
  const int n = (int)toks.size();
  std::vector<int64_t> comp(n);
  for (int i = 0; i < n; i++) comp[i] = (mask && !(*mask)[i]) ? -1 : p.token_map[toks[i]];
  std::vector<uint32_t> out((size_t)n * 48);
  for (int pos = 0; pos < n; pos++) {
    uint64_t c[4];
    bool blocked = false;
    for (int s = 0; s < 4; s++) {
      int64_t src = comp[pos - s < 0 ? 0 : pos - s];
      blocked = blocked || pos < s || src == -1;
      c[s] = blocked ? (uint64_t)p.pad_id : (uint64_t)src;
    }
    for (int l = 0; l < 2; l++) {
      uint64_t prod[4];
      for (int s = 0; s < 4; s++) prod[s] = c[s] * p.mult[l][s];
      uint64_t h = prod[0];
      for (int i = 1; i < 4; i++) {
        h ^= prod[i];
        for (int k = 0; k < 8; k++) {
          int col = l * 24 + (i - 1) * 8 + k;
          out[(size_t)pos * 48 + col] = (uint32_t)(h % p.prime[col] + p.offset[col]);
        }
      }
    }
  }
  return out;
}

static bool is_prime(uint64_t n) {
  if (n < 2) return false;
  for (uint64_t d = 2; d * d <= n; d++) if (n % d == 0) return false;
  return true;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "--dump")) {
    EngramHashParams p;
    if (!load_real(p)) { fprintf(stderr, "model metadata not available\n"); return 2; }
    std::vector<int32_t> toks;
    for (char* s = argv[2]; *s;) { toks.push_back((int32_t)strtol(s, &s, 10)); if (*s == ',') s++; }
    int split = argc > 3 ? atoi(argv[3]) : (int)toks.size();
    EngramHasher h(&p);
    std::vector<uint32_t> out(toks.size() * 48);
    if (split) h.hash(toks.data(), split, out.data());
    for (size_t i = split; i < toks.size(); i++) h.hash(&toks[i], 1, &out[i * 48]);
    for (size_t i = 0; i < toks.size(); i++) {
      printf("%zu", i);
      for (int c = 0; c < 48; c++) printf(" %u", out[i * 48 + c]);
      printf("\n");
    }
    return 0;
  }
  bool ok = true;
  std::mt19937_64 g(5);

  // ---- synthetic params ----
  {
    EngramHashParams p;
    std::vector<int64_t> tm(1000), lm(8), hv(48), ho(48);
    for (auto& v : tm) v = (int64_t)(g() % 700);
    for (auto& v : lm) v = (int64_t)((g() >> 20) | 1);
    int64_t cand = 1000, acc = 0;
    for (int i = 0; i < 48; i++) { do cand++; while (!is_prime((uint64_t)cand)); hv[i] = cand; ho[i] = acc; acc += cand; }
    ok &= p.init(tm, 2, lm, hv, ho);
    int bad = 0;
    for (int trial = 0; trial < 300; trial++) {
      int n = 1 + (int)(g() % 50);
      std::vector<int32_t> toks(n);
      std::vector<bool> mask(n, true);
      bool use_mask = trial % 3 == 0;
      for (int i = 0; i < n; i++) { toks[i] = (int32_t)(g() % 1000); if (use_mask) mask[i] = (g() % 5) != 0; }
      std::vector<uint32_t> ref = whole(p, toks, use_mask ? &mask : nullptr), got(n * 48), got2(n * 48);
      // random chunking (prefill chunks, then decode singles)
      EngramHasher h(&p);
      bool mk[64];
      for (int i = 0; i < n; i++) mk[i] = mask[i];
      for (int i = 0; i < n;) {
        int c = 1 + (int)(g() % 8);
        if (i + c > n) c = n - i;
        h.hash(&toks[i], c, &got[i * 48], use_mask ? mk + i : nullptr);
        i += c;
      }
      bad += got != ref;
      // hash_from leaves the state untouched; set_state round trip; set_history rebuild
      if (!use_mask && n > 4) {
        EngramHasher h2(&p);
        int k = n / 2;
        h2.hash(toks.data(), k, got2.data());
        EngramHasher::State st = h2.state();
        std::vector<uint32_t> tmp((n - k) * 48);
        h2.hash_from(st, &toks[k], n - k, tmp.data());
        bad += memcmp(&h2.state(), &st, sizeof st) != 0;
        EngramHasher h3(&p);
        h3.set_history(toks.data(), k, k);
        h3.hash(&toks[k], n - k, &got2[k * 48]);
        bad += memcmp(got2.data(), ref.data(), n * 48 * 4) != 0;
        bad += memcmp(tmp.data(), &ref[k * 48], (n - k) * 48 * 4) != 0;
      }
    }
    printf("synthetic: streaming/chunked/masked/state round-trips vs whole-sequence ref: %s (%d bad)\n",
           bad ? "FAIL" : "ok", bad);
    ok &= bad == 0;
  }

  // ---- real metadata ----
  EngramHashParams p;
  if (!load_real(p)) {
    printf("real metadata checks skipped (model not present)\n");
  } else {
    bool o = p.token_map.size() == 129280 && p.pad_id == 2;
    const uint64_t mult[2][4] = {{76632096046245ull, 4839876093313ull, 35959672319349ull, 73987337458391ull},
                                 {67716810739261ull, 51510806800915ull, 30921347202721ull, 82619226485591ull}};
    o &= !memcmp(p.mult, mult, sizeof mult);  // numpy default_rng(10007*layer) values (tools/engram_hash_check.py)
    uint64_t prev = 15999999, acc = 0;
    for (int i = 0; i < 48; i++) {  // consecutive distinct primes above 16M; offsets = global cumsum
      for (uint64_t c = prev + 1; c < p.prime[i]; c++) o &= !is_prime(c);
      o &= is_prime(p.prime[i]) && p.offset[i] == acc;
      acc += p.prime[i];
      prev = p.prime[i];
    }
    o &= acc == 768022850ull && p.offset[24] == 384006168ull;
    printf("real metadata (token_map, pad, multipliers, primes, offsets): %s\n", o ? "ok" : "FAIL");
    ok &= o;
    const int32_t toks[6] = {0, 9906, 1000, 128799, 2, 5};
    EngramHasher h(&p);
    uint32_t out[6 * 48];
    h.hash(toks, 3, out);
    h.hash(toks + 3, 1, out + 3 * 48);
    h.hash(toks + 4, 2, out + 4 * 48);
    bool gold = !memcmp(out, kGolden, sizeof out);
    printf("golden hashes (engram.py port, tools/engram_hash_check.py): %s\n", gold ? "ok" : "FAIL");
    ok &= gold;
  }
  printf(ok ? "test_arch_engram_hash: ALL OK\n" : "test_arch_engram_hash: FAILED\n");
  return ok ? 0 : 1;
}
