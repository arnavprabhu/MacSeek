#include "tok/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ms::json {

static const Value kNull;

Value::Value(long long v) : t_(Type::Number), is_int_(true), n_((double)v), s_(std::to_string(v)) {}
Value::Value(double v) : t_(Type::Number), is_int_(false), n_(v), s_(py_float_repr(v)) {}

Value Value::number_text(std::string text, double v, bool is_int) {
  Value r;
  r.t_ = Type::Number;
  r.n_ = v;
  r.is_int_ = is_int;
  r.s_ = std::move(text);
  return r;
}

bool Value::truthy() const {
  switch (t_) {
    case Type::Null: return false;
    case Type::Bool: return b_;
    case Type::Number: return n_ != 0;
    case Type::String: return !s_.empty();
    case Type::Array: return !a_.empty();
    case Type::Object: return !o_.empty();
  }
  return false;
}

const Value* Value::find(std::string_view k) const {
  if (t_ != Type::Object) return nullptr;
  for (auto& kv : o_)
    if (kv.first == k) return &kv.second;
  return nullptr;
}
Value* Value::find(std::string_view k) {
  if (t_ != Type::Object) return nullptr;
  for (auto& kv : o_)
    if (kv.first == k) return &kv.second;
  return nullptr;
}
const Value& Value::operator[](std::string_view k) const {
  const Value* v = find(k);
  return v ? *v : kNull;
}
void Value::set(std::string_view k, Value v) {
  if (t_ == Type::Null) t_ = Type::Object;
  if (Value* p = find(k)) { *p = std::move(v); return; }
  o_.emplace_back(std::string(k), std::move(v));
}
void Value::erase(std::string_view k) {
  for (size_t i = 0; i < o_.size(); i++)
    if (o_[i].first == k) { o_.erase(o_.begin() + (long)i); return; }
}

// ---------------- serializer ----------------

std::string py_float_repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round-trip
  std::string sci(buf, res.ptr);
  bool neg = sci[0] == '-';
  if (neg) sci.erase(0, 1);
  size_t epos = sci.find('e');
  std::string mant = sci.substr(0, epos);
  int exp10 = atoi(sci.c_str() + epos + 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits += c;
  int decpt = exp10 + 1;  // value = 0.digits * 10^decpt
  std::string out = neg ? "-" : "";
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (digits.size() > 1) { out += '.'; out += digits.substr(1); }
    char eb[16];
    snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    out += eb;
  } else if (decpt <= 0) {
    out += "0.";
    out.append((size_t)(-decpt), '0');
    out += digits;
  } else if ((size_t)decpt >= digits.size()) {
    out += digits;
    out.append((size_t)decpt - digits.size(), '0');
    out += ".0";
  } else {
    out += digits.substr(0, (size_t)decpt);
    out += '.';
    out += digits.substr((size_t)decpt);
  }
  return out;
}

void dump_string(std::string& out, std::string_view s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char b[8];
          snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out += (char)c;
        }
    }
  }
  out += '"';
}

void Value::dump_to(std::string& out) const {
  switch (t_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Number: out += s_; break;
    case Type::String: dump_string(out, s_); break;
    case Type::Array:
      out += '[';
      for (size_t i = 0; i < a_.size(); i++) {
        if (i) out += ", ";
        a_[i].dump_to(out);
      }
      out += ']';
      break;
    case Type::Object:
      out += '{';
      for (size_t i = 0; i < o_.size(); i++) {
        if (i) out += ", ";
        dump_string(out, o_[i].first);
        out += ": ";
        o_[i].second.dump_to(out);
      }
      out += '}';
      break;
  }
}

std::string Value::dump() const {
  std::string s;
  dump_to(s);
  return s;
}

// ---------------- parser ----------------

namespace {
struct Parser {
  const char* p;
  const char* b;
  const char* e;
  std::string err;
  int depth = 0;

  bool fail(const char* m) {
    if (err.empty()) err = std::string(m) + " at offset " + std::to_string(p - b);
    return false;
  }
  void ws() {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  }
  static void put_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 63)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
  }
  bool hex4(uint32_t& v) {
    if (e - p < 4) return fail("bad \\u escape");
    v = 0;
    for (int i = 0; i < 4; i++) {
      char c = *p++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
      else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
      else return fail("bad hex digit");
    }
    return true;
  }
  bool string(std::string& s) {
    p++;  // opening quote
    while (true) {
      const char* q = p;
      while (q < e && *q != '"' && *q != '\\') q++;
      s.append(p, q);
      p = q;
      if (p >= e) return fail("unterminated string");
      if (*p == '"') { p++; return true; }
      p++;  // backslash
      if (p >= e) return fail("bad escape");
      char c = *p++;
      switch (c) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          uint32_t cp;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp < 0xDC00 && e - p >= 6 && p[0] == '\\' && p[1] == 'u') {
            const char* save = p;
            p += 2;
            uint32_t lo;
            if (!hex4(lo)) return false;
            if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else p = save;
          }
          put_utf8(s, cp);
          break;
        }
        default: return fail("bad escape");
      }
    }
  }
  bool number(Value& out) {
    const char* s = p;
    bool is_int = true;
    if (p < e && *p == '-') p++;
    if (e - p >= 8 && memcmp(p, "Infinity", 8) == 0) {
      p += 8;
      out = Value(*s == '-' ? -INFINITY : INFINITY);
      return true;
    }
    if (p >= e || !(*p >= '0' && *p <= '9')) return fail("bad number");
    while (p < e && *p >= '0' && *p <= '9') p++;
    if (p < e && *p == '.') {
      is_int = false;
      p++;
      while (p < e && *p >= '0' && *p <= '9') p++;
    }
    if (p < e && (*p == 'e' || *p == 'E')) {
      is_int = false;
      p++;
      if (p < e && (*p == '+' || *p == '-')) p++;
      while (p < e && *p >= '0' && *p <= '9') p++;
    }
    std::string text(s, p);
    double v = strtod(text.c_str(), nullptr);
    if (is_int) {
      // Python int(): strip "-0" sign and keep arbitrary precision text
      size_t i = text[0] == '-' ? 1 : 0;
      bool all_zero = text.find_first_not_of('0', i) == std::string::npos;
      if (all_zero) text = "0";
      out = Value::number_text(text, v, true);
    } else {
      out = Value::number_text(py_float_repr(v), v, false);
    }
    return true;
  }
  bool value(Value& out) {
    ws();
    if (p >= e) return fail("unexpected end");
    if (++depth > 512) return fail("nesting too deep");
    bool ok = true;
    char c = *p;
    if (c == '{') {
      p++;
      out = Value::object();
      ws();
      if (p < e && *p == '}') { p++; }
      else {
        while (true) {
          ws();
          if (p >= e || *p != '"') { ok = fail("expected key"); break; }
          std::string k;
          if (!string(k)) { ok = false; break; }
          ws();
          if (p >= e || *p != ':') { ok = fail("expected ':'"); break; }
          p++;
          Value v;
          if (!value(v)) { ok = false; break; }
          out.set(k, std::move(v));  // duplicate keys: last wins (Python)
          ws();
          if (p < e && *p == ',') { p++; continue; }
          if (p < e && *p == '}') { p++; break; }
          ok = fail("expected ',' or '}'");
          break;
        }
      }
    } else if (c == '[') {
      p++;
      out = Value::array();
      ws();
      if (p < e && *p == ']') { p++; }
      else {
        while (true) {
          Value v;
          if (!value(v)) { ok = false; break; }
          out.push_back(std::move(v));
          ws();
          if (p < e && *p == ',') { p++; continue; }
          if (p < e && *p == ']') { p++; break; }
          ok = fail("expected ',' or ']'");
          break;
        }
      }
    } else if (c == '"') {
      std::string s;
      ok = string(s);
      out = Value(std::move(s));
    } else if (c == 't' && e - p >= 4 && memcmp(p, "true", 4) == 0) { p += 4; out = Value(true); }
    else if (c == 'f' && e - p >= 5 && memcmp(p, "false", 5) == 0) { p += 5; out = Value(false); }
    else if (c == 'n' && e - p >= 4 && memcmp(p, "null", 4) == 0) { p += 4; out = Value(); }
    else if (c == 'N' && e - p >= 3 && memcmp(p, "NaN", 3) == 0) { p += 3; out = Value(NAN); }
    else if (c == '-' || c == 'I' || (c >= '0' && c <= '9')) {
      if (c == 'I') {
        if (e - p >= 8 && memcmp(p, "Infinity", 8) == 0) { p += 8; out = Value(INFINITY); }
        else ok = fail("bad literal");
      } else {
        ok = number(out);
      }
    } else {
      ok = fail("unexpected character");
    }
    depth--;
    return ok;
  }
};
}  // namespace

bool parse(std::string_view text, Value& out, std::string* err) {
  Parser ps{text.data(), text.data(), text.data() + text.size(), {}};
  bool ok = ps.value(out);
  if (ok) {
    ps.ws();
    if (ps.p != ps.e) ok = ps.fail("trailing characters");
  }
  if (!ok && err) *err = ps.err;
  return ok;
}

Value parse_or_null(std::string_view text) {
  Value v;
  if (!parse(text, v)) return Value();
  return v;
}

}  // namespace ms::json
