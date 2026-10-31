#pragma once
// Minimal JSON value + parser + serializer (dependency-free).
// - Objects keep insertion order (needed: tool schemas are re-serialized in the caller's key order).
// - dump() reproduces Python's json.dumps(v, ensure_ascii=False) byte-for-byte (", " / ": " separators,
//   Python float repr, \uXXXX only for control chars), which is what DeepSeek's encoding.py uses.
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ms::json {

enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

class Value {
 public:
  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : t_(Type::Bool), b_(b) {}
  Value(int v) : Value((long long)v) {}
  Value(long v) : Value((long long)v) {}
  Value(long long v);
  Value(double v);
  Value(const char* s) : t_(Type::String), s_(s) {}
  Value(std::string s) : t_(Type::String), s_(std::move(s)) {}
  Value(std::string_view s) : t_(Type::String), s_(s) {}
  static Value array() { Value v; v.t_ = Type::Array; return v; }
  static Value object() { Value v; v.t_ = Type::Object; return v; }
  // Number with explicit Python-style text (e.g. integer literal kept verbatim).
  static Value number_text(std::string text, double v, bool is_int);

  Type type() const { return t_; }
  bool is_null() const { return t_ == Type::Null; }
  bool is_bool() const { return t_ == Type::Bool; }
  bool is_number() const { return t_ == Type::Number; }
  bool is_int() const { return t_ == Type::Number && is_int_; }
  bool is_string() const { return t_ == Type::String; }
  bool is_array() const { return t_ == Type::Array; }
  bool is_object() const { return t_ == Type::Object; }
  // Python truthiness (None/False/0/""/[]/{} are falsy).
  bool truthy() const;

  bool as_bool(bool d = false) const { return t_ == Type::Bool ? b_ : d; }
  double as_double(double d = 0) const { return t_ == Type::Number ? n_ : d; }
  long long as_int(long long d = 0) const { return t_ == Type::Number ? (long long)n_ : d; }
  const std::string& str() const { return s_; }  // string value (empty if not a string)
  std::string as_string(const std::string& d = "") const { return t_ == Type::String ? s_ : d; }
  const std::string& num_text() const { return s_; }

  // arrays
  const Array& arr() const { return a_; }
  Array& arr() { return a_; }
  size_t size() const { return t_ == Type::Array ? a_.size() : t_ == Type::Object ? o_.size() : 0; }
  const Value& operator[](size_t i) const { return a_[i]; }
  void push_back(Value v) { if (t_ == Type::Null) t_ = Type::Array; a_.push_back(std::move(v)); }

  // objects
  const Object& obj() const { return o_; }
  Object& obj() { return o_; }
  const Value* find(std::string_view k) const;  // nullptr if missing (or not an object)
  Value* find(std::string_view k);
  bool has(std::string_view k) const { return find(k) != nullptr; }
  // missing key -> shared null value
  const Value& operator[](std::string_view k) const;
  const Value& get(std::string_view k) const { return (*this)[k]; }
  void set(std::string_view k, Value v);  // replace in place or append
  void erase(std::string_view k);

  std::string dump() const;  // Python json.dumps(ensure_ascii=False) compatible
  void dump_to(std::string& out) const;

 private:
  Type t_ = Type::Null;
  bool b_ = false;
  bool is_int_ = false;
  double n_ = 0;
  std::string s_;  // string value, or number text
  Array a_;
  Object o_;
};

// Parse JSON text. On error returns false and sets *err (with byte offset).
bool parse(std::string_view text, Value& out, std::string* err = nullptr);
Value parse_or_null(std::string_view text);

// Python repr() of a float, e.g. 1.0, 0.1, 1e-05, 1e+16, inf.
std::string py_float_repr(double v);
// JSON string literal (with quotes) in Python ensure_ascii=False style.
void dump_string(std::string& out, std::string_view s);

}  // namespace ms::json
