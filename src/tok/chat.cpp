#include "tok/chat.hpp"

#include <stdexcept>
#include <vector>

namespace ms::chat {

using json::Value;

namespace {

const char* kUser = "<｜User｜>";
const char* kAssistant = "<｜Assistant｜>";
const char* kSystem = "<｜System｜>";
const char* kLatestReminder = "<｜latest_reminder｜>";
const char* kImage = "<｜deepseek_image｜>";

struct Err : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::string str_of(const Value& v) {
  if (v.is_string()) return v.str();
  if (v.is_null()) return "";
  return v.dump();
}

bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

std::string join(const std::vector<std::string>& parts, const char* sep) {
  std::string out;
  for (size_t i = 0; i < parts.size(); i++) {
    if (i) out += sep;
    out += parts[i];
  }
  return out;
}

const std::string& role_of(const Value& m) {
  static const std::string empty;
  const Value* r = m.find("role");
  return r && r->is_string() ? r->str() : empty;
}

// ---- tool names ----
std::pair<Value, std::string> split_tool_name(const std::string& name, const Value& ns_in) {
  Value ns = ns_in;
  std::string bare = name;
  size_t p = name.find("::");
  if (p != std::string::npos) {
    std::string prefix = name.substr(0, p);
    if (!ns.is_null() && !(ns.is_string() && ns.str() == prefix))
      throw Err("Conflicting tool namespaces: " + str_of(ns) + " != " + prefix);
    ns = Value(prefix);
    bare = name.substr(p + 2);
  }
  if (bare.find("::") != std::string::npos) throw Err("Tool name must not contain '::': " + bare);
  if (ns.is_string() && ns.str().find("::") != std::string::npos)
    throw Err("Tool namespace must not contain '::': " + ns.str());
  return {ns, bare};
}

std::string tool_name_for_encoding(const Value& tool) {
  Value ns = tool["namespace"];
  if (ns.is_object()) ns = ns["name"];
  auto [n2, name] = split_tool_name(str_of(tool["name"]), ns);
  return n2.is_null() ? name : str_of(n2) + "::" + name;
}

Value tools_from_openai_format(const Value& tools) {
  Value fns = Value::array();
  for (const Value& tool : tools.arr()) {
    Value fn = tool.has("function") ? tool["function"] : tool;  // template also accepts bare definitions
    if (!fn.is_object()) throw Err("tool definition must be an object");
    if (tool.has("function") && !tool["namespace"].is_null()) fn.set("namespace", tool["namespace"]);
    fn.set("name", Value(tool_name_for_encoding(fn)));
    Value ns = fn["namespace"];
    fn.erase("namespace");
    if (ns.is_object() && ns["description"].truthy())
      fn.set("description", Value(str_of(ns["description"]) + "\n" + str_of(fn["description"])));
    fns.push_back(std::move(fn));
  }
  return fns;
}

std::string render_tools(const Value& fns) {
  std::vector<std::string> schemas;
  for (const Value& f : fns.arr()) schemas.push_back(f.dump());
  std::string s =
      "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by "
      "writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke "
      "name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" "
      "string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke "
      "name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified "
      "as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in "
      "JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST "
      "output your complete reasoning inside <think>...</think> BEFORE any tool calls or final "
      "response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available "
      "Tool Schemas\n\n";
  s += join(schemas, "\n");
  s += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
  return s;
}

// OpenAI tool_calls -> internal {name, arguments, namespace?}
Value tool_calls_from_openai_format(const Value& tcs) {
  Value calls = Value::array();
  for (const Value& tc : tcs.arr()) {
    const Value& fn = tc.has("function") ? tc["function"] : tc;
    Value ns = tc["namespace"].truthy() ? tc["namespace"] : fn["namespace"];
    auto [n2, name] = split_tool_name(str_of(fn["name"]), ns.truthy() ? ns : Value());
    Value call = Value::object();
    call.set("name", Value(name));
    call.set("arguments", fn["arguments"]);
    if (!n2.is_null()) call.set("namespace", n2);
    calls.push_back(std::move(call));
  }
  return calls;
}

std::string encode_arguments_to_dsml(const Value& call) {
  Value args = call["arguments"];
  if (!args.is_object()) {
    for (int k = 0; k < 2; k++) {  // tolerate JSON strings, including double-encoded ones
      if (!args.is_string()) break;
      Value parsed;
      if (!json::parse(args.str(), parsed)) break;
      args = std::move(parsed);
    }
    if (!args.is_object()) {
      Value w = Value::object();
      w.set("arguments", call["arguments"]);
      args = std::move(w);
    }
  }
  std::vector<std::string> lines;
  for (const auto& [k, v] : args.obj()) {
    std::string line = std::string("<") + kDsml + " parameter name=\"" + k + "\" string=\"" +
                       (v.is_string() ? "true" : "false") + "\">" + (v.is_string() ? v.str() : v.dump()) + "</" +
                       kDsml + " parameter>";
    lines.push_back(std::move(line));
  }
  return join(lines, "\n");
}

// ---- preprocessing (encoding.py: merge_tool_messages, sort_tool_results_by_call_order, process_image_messages)
Value text_block(const std::string& t) {
  Value b = Value::object();
  b.set("type", Value("text"));
  b.set("text", Value(t));
  return b;
}

std::vector<Value> merge_tool_messages(const Value& messages) {
  std::vector<Value> merged;
  for (const Value& m0 : messages.arr()) {
    if (!m0.is_object()) throw Err("message must be an object");
    Value msg = m0;
    const std::string role = role_of(msg);
    if (role == "tool") {
      Value content = msg.has("content") ? msg["content"] : Value("");
      if (msg["content_blocks"].truthy() && !msg["content"].is_string()) content = msg["content_blocks"];
      Value tb = Value::object();
      tb.set("type", Value("tool_result"));
      tb.set("tool_use_id", msg.has("tool_call_id") ? msg["tool_call_id"] : Value(""));
      tb.set("content", content);
      if (!merged.empty() && role_of(merged.back()) == "user" && merged.back().has("content_blocks")) {
        merged.back().find("content_blocks")->push_back(std::move(tb));
      } else {
        Value u = Value::object();
        u.set("role", Value("user"));
        Value blocks = Value::array();
        blocks.push_back(std::move(tb));
        u.set("content_blocks", std::move(blocks));
        merged.push_back(std::move(u));
      }
    } else if (role == "user") {
      Value blocks = msg["content_blocks"];
      if (blocks.is_null()) {
        Value content = msg.has("content") ? msg["content"] : Value("");
        if (content.is_array()) blocks = content;
        else {
          blocks = Value::array();
          blocks.push_back(text_block(str_of(content)));
        }
      }
      if (!merged.empty() && role_of(merged.back()) == "user" && merged.back().has("content_blocks") &&
          merged.back()["task"].is_null()) {
        Value* dst = merged.back().find("content_blocks");
        for (const Value& b : blocks.arr()) dst->push_back(b);
      } else {
        msg.set("content_blocks", std::move(blocks));
        merged.push_back(std::move(msg));
      }
    } else {
      merged.push_back(std::move(msg));
    }
  }
  return merged;
}

void sort_tool_results(std::vector<Value>& msgs) {
  std::vector<std::pair<std::string, int>> order;  // last assistant's tool_call ids
  for (Value& m : msgs) {
    const std::string role = role_of(m);
    if (role == "assistant" && m["tool_calls"].truthy()) {
      order.clear();
      int idx = 0;
      for (const Value& tc : m["tool_calls"].arr()) {
        std::string id = str_of(tc["id"]);
        if (id.empty()) id = str_of(tc["function"]["id"]);
        if (!id.empty()) order.emplace_back(id, idx);
        idx++;
      }
    } else if (role == "user" && m["content_blocks"].truthy()) {
      Value* blocks = m.find("content_blocks");
      std::vector<Value> tool_blocks;
      for (const Value& b : blocks->arr())
        if (str_of(b["type"]) == "tool_result") tool_blocks.push_back(b);
      if (tool_blocks.size() > 1 && !order.empty()) {
        auto key = [&](const Value& b) {
          std::string id = str_of(b["tool_use_id"]);
          for (auto& [k, i] : order)
            if (k == id) return i;  // NB python dict: last duplicate wins; ids are unique in practice
          return 0;
        };
        std::stable_sort(tool_blocks.begin(), tool_blocks.end(),
                         [&](const Value& a, const Value& b) { return key(a) < key(b); });
        size_t si = 0;
        for (Value& b : blocks->arr())
          if (str_of(b["type"]) == "tool_result") b = tool_blocks[si++];
      }
    }
  }
}

Value process_image_blocks(const Value& blocks) {
  Value out = Value::array();
  for (const Value& b : blocks.arr()) {
    if (!b.is_object()) { out.push_back(b); continue; }
    std::string type = str_of(b["type"]);
    if (type == "image" || type == "image_url") {
      out.push_back(text_block(kImage));
    } else if (type == "tool_result" && b["content"].is_array()) {
      Value nb = b;
      nb.set("content", process_image_blocks(b["content"]));
      out.push_back(std::move(nb));
    } else if (type == "text") {
      if (contains(str_of(b["text"]), kImage))
        throw Err("Text block contains image placeholder. Images should be separate content blocks.");
      out.push_back(b);
    } else {
      out.push_back(b);
    }
  }
  return out;
}

void validate_no_image_tokens(const Value& m) {
  if (m["content"].is_string() && contains(m["content"].str(), kImage))
    throw Err("Message content contains image special token <｜deepseek_image｜>. Images should be image blocks.");
  for (const char* k : {"reasoning_content", "thinking"})
    if (m[k].is_string() && contains(m[k].str(), kImage)) throw Err("reasoning_content contains image special token");
}

void process_image_messages(std::vector<Value>& msgs) {
  for (Value& m : msgs) {
    validate_no_image_tokens(m);
    if (m["content"].is_array() && !m.has("content_blocks")) {
      m.set("content_blocks", m["content"]);
      m.erase("content");
    }
    if (m["content_blocks"].truthy()) {
      m.set("content_blocks", process_image_blocks(m["content_blocks"]));
      if (!m["content"].is_string()) {
        std::vector<std::string> texts;
        for (const Value& b : m["content_blocks"].arr())
          if (b.is_object() && str_of(b["type"]) == "text") texts.push_back(str_of(b["text"]));
        m.set("content", Value(join(texts, "\n\n")));
      }
    }
  }
}

// ---- rendering ----
int find_last_user_index(const std::vector<Value>& msgs) {
  for (int i = (int)msgs.size() - 1; i >= 0; i--) {
    const std::string& role = role_of(msgs[(size_t)i]);
    if (role == "user" || (role == "system" && i > 0)) {
      const Value& cb = msgs[(size_t)i]["content_blocks"];
      if (cb.truthy()) {
        bool any = false;
        for (const Value& b : cb.arr()) {
          std::string t = str_of(b["type"]);
          if (t == "text" || t == "image" || t == "image_url") any = true;
        }
        if (!any) continue;
      }
      return i;
    }
  }
  return -1;
}

const Value& reasoning_of(const Value& m) {
  const Value& rc = m["reasoning_content"];
  return rc.is_null() ? m["thinking"] : rc;
}

std::string task_token(const std::string& task) {
  static const std::pair<const char*, const char*> kTasks[] = {
      {"action", "<｜action｜>"}, {"query", "<｜query｜>"}, {"authority", "<｜authority｜>"},
      {"domain", "<｜domain｜>"}, {"title", "<｜title｜>"},  {"read_url", "<｜read_url｜>"}};
  for (auto& [k, v] : kTasks)
    if (task == k) return v;
  throw Err("Invalid task: '" + task + "'");
}

std::string render_message(size_t index, const std::vector<Value>& msgs, bool thinking, bool drop_thinking,
                           int budget, bool add_generation_prompt) {
  const Value& msg = msgs[index];
  const int last_user = find_last_user_index(msgs);
  const std::string role = role_of(msg);
  std::string prompt;
  const bool effort_prompt = index == 0 && thinking;
  if (index == 0 && (effort_prompt || role == "system")) prompt += kSystem;
  if (effort_prompt)
    prompt += "Reasoning Effort: " + std::to_string(budget) +
              " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

  if (role == "system") {
    if (index > 0) prompt += kSystem;
    prompt += str_of(msg["content"]);
    if (msg["tools"].truthy()) prompt += "\n\n" + render_tools(tools_from_openai_format(msg["tools"]));
    if (msg["response_format"].truthy())
      prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" +
                msg["response_format"].dump();
  } else if (role == "user") {
    prompt += kUser;
    const Value& cb = msg["content_blocks"];
    if (cb.truthy()) {
      std::vector<std::string> parts;
      for (const Value& b : cb.arr()) {
        std::string type = str_of(b["type"]);
        if (type == "text") {
          parts.push_back(str_of(b["text"]));
        } else if (type == "tool_result") {
          const Value& tc = b["content"];
          std::string tcs;
          if (tc.is_array()) {
            std::vector<std::string> sub;
            for (const Value& s : tc.arr()) {
              if (str_of(s["type"]) == "text") sub.push_back(str_of(s["text"]));
              else sub.push_back("[Unsupported " + str_of(s["type"]) + "]");
            }
            tcs = join(sub, "\n\n");
          } else {
            tcs = str_of(tc);
          }
          parts.push_back("<tool_result>" + tcs + "</tool_result>");
        } else {
          parts.push_back("[Unsupported " + type + "]");
        }
      }
      prompt += join(parts, "\n\n");
    } else {
      prompt += str_of(msg["content"]);
    }
  } else if (role == "latest_reminder") {
    prompt += kLatestReminder + str_of(msg["content"]);
  } else if (role == "tool") {
    throw Err("tool messages must be merged into user messages");
  } else if (role == "assistant") {
    std::string thinking_part, tc_content;
    if (msg["tool_calls"].truthy()) {
      Value calls = tool_calls_from_openai_format(msg["tool_calls"]);
      std::vector<std::string> tcl;
      for (const Value& tc : calls.arr())
        tcl.push_back(std::string("<") + kDsml + " invoke name=\"" + tool_name_for_encoding(tc) + "\">\n" +
                      encode_arguments_to_dsml(tc) + "\n</" + kDsml + " invoke>");
      tc_content += std::string("\n\n<") + kDsml + " calls>\n" + join(tcl, "\n") + "\n</" + kDsml + " calls>";
    }
    const bool prev_has_task = index >= 1 && !msgs[index - 1]["task"].is_null();
    if (thinking && !prev_has_task) {
      if (!drop_thinking || (int)index > last_user) thinking_part = str_of(reasoning_of(msg)) + kThinkEnd;
    }
    prompt += thinking_part + str_of(msg["content"]) + tc_content;
    if (!msg["wo_eos"].truthy()) prompt += kEos;
  } else {
    throw Err("Unknown role: " + role);
  }

  // transition tokens
  if (index + 1 < msgs.size()) {
    const std::string& next = role_of(msgs[index + 1]);
    if (next != "assistant" && next != "latest_reminder") return prompt;
  }
  const Value& task = msg["task"];
  if (!task.is_null()) {
    std::string t = str_of(task);
    std::string tok = task_token(t);
    if (t != "action") {
      prompt += tok;
    } else {
      prompt += kAssistant;
      prompt += thinking ? kThinkStart : kThinkEnd;
      prompt += tok;
    }
  } else if (role == "user" || (role == "system" && index > 0)) {
    const bool is_last = index + 1 == msgs.size();
    if (!is_last || add_generation_prompt) {
      prompt += kAssistant;
      if (!drop_thinking && thinking) prompt += kThinkStart;
      else if (drop_thinking && thinking && (int)index >= last_user) prompt += kThinkStart;
      else prompt += kThinkEnd;
    }
  }
  return prompt;
}

std::vector<Value> drop_thinking_messages(const std::vector<Value>& msgs) {
  const int last_user = find_last_user_index(msgs);
  std::vector<Value> out;
  for (size_t i = 0; i < msgs.size(); i++) {
    const std::string& role = role_of(msgs[i]);
    if (role == "user" || role == "system" || role == "tool" || role == "latest_reminder" ||
        role == "direct_search_results" || (int)i >= last_user) {
      out.push_back(msgs[i]);
    } else if (role == "assistant") {
      Value m = msgs[i];
      m.erase("reasoning_content");
      m.erase("thinking");
      out.push_back(std::move(m));
    }
  }
  return out;
}

}  // namespace

int effort_budget(const Value& e) {
  if (e.is_null()) return 75;
  if (e.is_string()) {
    const std::string& s = e.str();
    if (s == "low") return 50;
    if (s == "medium") return 62;
    if (s == "high") return 75;
    if (s == "max") return 100;
    return -1;
  }
  if (e.is_int()) {
    long long v = e.as_int();
    return (v >= 1 && v <= 100) ? (int)v : -1;
  }
  return -1;
}

bool format(const Value& messages, const Value& tools, const Value& response_format, const Options& opt,
            std::string& out, std::string* err) {
  try {
    if (!messages.is_array()) throw Err("messages must be an array");
    if (opt.thinking_mode != "thinking" && opt.thinking_mode != "chat")
      throw Err("Invalid thinking_mode `" + opt.thinking_mode + "`, should be \"chat\" or \"thinking\"");
    const int budget = effort_budget(opt.reasoning_effort);
    if (budget < 0)
      throw Err("Invalid reasoning effort: " + str_of(opt.reasoning_effort) +
                ", should be int within [1,100] or one of \"low\", \"medium\", \"high\", \"max\"");
    const bool thinking = opt.thinking_mode == "thinking";

    for (const Value& m : messages.arr()) validate_no_image_tokens(m);
    std::vector<Value> msgs = merge_tool_messages(messages);
    // top-level tools / response_format -> leading system message (GGUF template behaviour)
    if (tools.truthy() || response_format.truthy()) {
      if (msgs.empty() || role_of(msgs[0]) != "system") {
        Value sys = Value::object();
        sys.set("role", Value("system"));
        sys.set("content", Value(""));
        msgs.insert(msgs.begin(), std::move(sys));
      }
      if (tools.truthy() && !msgs[0]["tools"].truthy()) msgs[0].set("tools", tools);
      if (response_format.truthy() && !msgs[0]["response_format"].truthy())
        msgs[0].set("response_format", response_format);
    }
    sort_tool_results(msgs);
    process_image_messages(msgs);

    out.clear();
    if (opt.add_bos) out += kBos;
    bool drop = opt.drop_thinking;
    for (const Value& m : msgs)
      if (m["tools"].truthy()) drop = false;
    if (thinking && drop) msgs = drop_thinking_messages(msgs);
    for (size_t i = 0; i < msgs.size(); i++)
      out += render_message(i, msgs, thinking, drop, budget, opt.add_generation_prompt);
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

bool format_request(const Value& body, std::string& out, std::string* err, Options* resolved) {
  Options opt;
  const Value& kw = body["chat_template_kwargs"];
  auto get = [&](const char* k) -> const Value& {
    const Value& v = kw[k];
    return v.is_null() ? body[k] : v;
  };
  const Value& tm = get("thinking_mode");
  const Value& et = get("enable_thinking");
  const Value& th = get("thinking");
  if (tm.is_string()) opt.thinking_mode = tm.str();
  else if (et.is_bool()) opt.thinking_mode = et.as_bool() ? "thinking" : "chat";
  else if (th.is_bool()) opt.thinking_mode = th.as_bool() ? "thinking" : "chat";
  opt.reasoning_effort = get("reasoning_effort");
  if (get("drop_thinking").is_bool()) opt.drop_thinking = get("drop_thinking").as_bool();
  if (get("add_generation_prompt").is_bool()) opt.add_generation_prompt = get("add_generation_prompt").as_bool();
  if (resolved) *resolved = opt;
  return format(body["messages"], body["tools"], body["response_format"], opt, out, err);
}

// ---------------------------------------------------------------------------------------------------------------
// Reply parsing (inverse of the assistant rendering; mirrors encoding.py parse_message_from_completion_text)
// ---------------------------------------------------------------------------------------------------------------

namespace {
bool parse_tool_calls(std::string_view text, Value& calls) {
  const std::string calls_end = std::string("</") + kDsml + " calls>";
  const std::string invoke_start = std::string("<") + kDsml + " invoke";
  const std::string invoke_end = std::string("</") + kDsml + " invoke>";
  const std::string param_start = std::string("<") + kDsml + " parameter";
  const std::string param_end = std::string("</") + kDsml + " parameter>";
  size_t i = 0;
  auto skip_nl = [&]() { while (i < text.size() && text[i] == '\n') i++; };
  while (true) {
    skip_nl();
    if (text.compare(i, calls_end.size(), calls_end) == 0) return true;
    if (text.compare(i, invoke_start.size(), invoke_start) != 0) return false;
    i += invoke_start.size();
    const std::string name_pfx = " name=\"";
    if (text.compare(i, name_pfx.size(), name_pfx) != 0) return false;
    i += name_pfx.size();
    size_t q = text.find("\">", i);
    if (q == std::string_view::npos) return false;
    std::string name(text.substr(i, q - i));
    i = q + 2;
    std::string args = "{";
    bool first = true;
    while (true) {
      skip_nl();
      if (text.compare(i, invoke_end.size(), invoke_end) == 0) { i += invoke_end.size(); break; }
      if (text.compare(i, param_start.size(), param_start) != 0) return false;
      i += param_start.size();
      if (text.compare(i, name_pfx.size(), name_pfx) != 0) return false;
      i += name_pfx.size();
      size_t q2 = text.find('"', i);
      if (q2 == std::string_view::npos) return false;
      std::string pname(text.substr(i, q2 - i));
      i = q2 + 1;
      const std::string s_true = " string=\"true\">", s_false = " string=\"false\">";
      bool is_str;
      if (text.compare(i, s_true.size(), s_true) == 0) { is_str = true; i += s_true.size(); }
      else if (text.compare(i, s_false.size(), s_false) == 0) { is_str = false; i += s_false.size(); }
      else return false;
      size_t q3 = text.find(param_end, i);
      if (q3 == std::string_view::npos) return false;
      std::string val(text.substr(i, q3 - i));
      i = q3 + param_end.size();
      if (!first) args += ", ";
      first = false;
      json::dump_string(args, pname);
      args += ": ";
      if (is_str) json::dump_string(args, val);
      else args += val;
    }
    args += "}";
    Value call = Value::object();
    call.set("type", Value("function"));
    Value fn = Value::object();
    fn.set("name", Value(name));
    fn.set("arguments", Value(args));
    call.set("function", std::move(fn));
    calls.push_back(std::move(call));
  }
}
}  // namespace

Reply parse_reply(std::string_view text, bool thinking) {
  Reply r;
  size_t i = 0;
  if (thinking) {
    // the prompt already ends with <think>; tolerate a repeated one
    if (text.substr(0, 7) == kThinkStart) i = 7;
    size_t e = text.find(kThinkEnd, i);
    if (e == std::string_view::npos) { r.reasoning = std::string(text.substr(i)); return r; }
    r.reasoning = std::string(text.substr(i, e - i));
    i = e + 8;
  }
  std::string_view rest = text.substr(i);
  size_t eos = rest.find(kEos);
  if (eos != std::string_view::npos) rest = rest.substr(0, eos);
  const std::string tc_start = std::string("\n\n<") + kDsml + " calls>";
  size_t t = rest.find(tc_start);
  if (t != std::string_view::npos) {
    Value calls = Value::array();
    if (parse_tool_calls(rest.substr(t + tc_start.size()), calls)) {
      r.content = std::string(rest.substr(0, t));
      r.tool_calls = std::move(calls);
      return r;
    }
  }
  r.content = std::string(rest);
  return r;
}

}  // namespace ms::chat
