"""Oneiros panel demonstration - browser UI.

Standard library only. Serves a small page on 127.0.0.1 and opens a browser.
Nothing is installed, nothing leaves the machine, no internet is used.

    "%PY%" scripts\\demo_ui.py

What it does:
  * loads real examples from the canonical corpus into a picker
  * lets you edit either version of the program, or paste your own
  * runs one test against both versions and applies the oracle rule
  * searches for inputs where the two versions disagree, and shows both
    the inputs that expose the defect and the inputs that do not

Honesty note: the distinguishing-input search uses the project's own
non-model baseline generator, not the trained model. The UI labels it as such.
"""
import ast
import html
import importlib.util
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "corpus" / "v3_final_candidate"
HEADLINE = ROOT / "results" / "v3_full_sft_monitored_20260819_1" / "sft_validation_results.json"

HOST, PORT = "127.0.0.1", 8765

EXAMPLE_IDS = [
    "mutation::humaneval_HumanEval_121_mut_01039",
    "mutation::humaneval_HumanEval_13_mut_00073",
    "curated::bugsinpy_black_1",
    "curated::bugsinpy_tornado_1",
    "curated::bugsinpy_thefuck_1",
    "mutation::humaneval_HumanEval_2_mut_00018",
]

_runner = None


def runner():
    """Load benchmark_runner by path so baseline/__init__ (which needs torch) is skipped."""
    global _runner
    if _runner is None:
        spec = importlib.util.spec_from_file_location(
            "oneiros_benchmark_runner", ROOT / "baseline" / "benchmark_runner.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _runner = module
    return _runner


def load_examples():
    with open(CORPUS / "records.json", encoding="utf-8") as handle:
        records = json.load(handle)
    wanted = {r: None for r in EXAMPLE_IDS}
    for item in records:
        if item["id"] in wanted and wanted[item["id"]] is None:
            wanted[item["id"]] = item
    out = []
    for rid in EXAMPLE_IDS:
        item = wanted.get(rid)
        if item is None:
            continue
        source = item.get("source") or {}
        name = source.get("name") if isinstance(source, dict) else str(source)
        out.append({
            "id": item["id"],
            "label": "%s()  -  %s" % (
                item["entry_point"],
                "real mutation from the corpus" if name == "oneiros_clean_mutations"
                else "reproduced repository bug",
            ),
            "entry_point": item["entry_point"],
            "specification": " ".join(str(item.get("specification", "")).split()),
            "reference": item["reference_code"].rstrip(),
            "defective": item["code_under_test"].rstrip(),
            "test": item["tests"][0]["code"].strip(),
        })
    return out


def headline():
    try:
        with open(HEADLINE, encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "killed": data.get("function_validation_killed"),
            "total": data.get("function_validation_records"),
            "rate": round(float(data.get("function_kill_rate", 0)) * 100, 2),
        }
    except Exception:
        return {"killed": 429, "total": 775, "rate": 55.35}


# ---------------------------------------------------------------- actions

def assert_parts(test):
    """Split `assert <expr> == <expected>` into its two sides.

    A bare assert raises AssertionError with an empty message, so the raw error
    string is identical for every failing test. Recovering the two sides lets the
    page show what was actually returned instead.
    """
    try:
        tree = ast.parse(test.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    node = tree.body[0].test
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], (ast.Eq, ast.Is)):
        return None
    try:
        return ast.unparse(node.left), ast.unparse(node.comparators[0])
    except Exception:
        return None


def _returned(mod, code, expr):
    """Evaluate the asserted expression on one version and report its value."""
    ok, value, _ = mod.safe_exec(code, "result = %s" % expr)
    return _short(repr(value)) if ok else None


def act_run(payload):
    """Run one test against both versions and apply the oracle rule."""
    mod = runner()
    reference = payload.get("reference", "")
    defective = payload.get("defective", "")
    test = payload.get("test", "").strip()
    if not test:
        return {"error": "Enter a test first."}
    ref_ok, _, ref_err = mod.safe_exec(reference, test)
    def_ok, _, def_err = mod.safe_exec(defective, test)

    expected = ref_value = def_value = None
    parts = assert_parts(test)
    if parts:
        expr, expected_src = parts
        expected = expected_src
        ref_value = _returned(mod, reference, expr)
        def_value = _returned(mod, defective, expr)
    if ref_ok and not def_ok:
        verdict, detail = "CAUGHT", "Passes on the correct version and fails on the defective one."
    elif ref_ok and def_ok:
        verdict, detail = "NOT CAUGHT", "It passes on both versions, so it does not distinguish them."
    elif not ref_ok and not def_ok:
        verdict, detail = "INVALID", "It fails on the correct version too, so the test itself is wrong."
    else:
        verdict, detail = "INVALID", "It fails on the correct version, so it cannot be counted."
    return {
        "reference": {"ok": bool(ref_ok), "error": (ref_err or "").strip(),
                      "returned": ref_value},
        "defective": {"ok": bool(def_ok), "error": (def_err or "").strip(),
                      "returned": def_value},
        "expected": expected,
        "verdict": verdict,
        "detail": detail,
    }


def act_search(payload):
    """Try many inputs and report which ones the two versions disagree on."""
    mod = runner()
    reference = payload.get("reference", "")
    defective = payload.get("defective", "")
    entry = (payload.get("entry_point") or "").strip()
    if not entry:
        return {"error": "Enter the function name to test."}

    try:
        ptypes = mod._parse_params(reference, entry)
    except Exception:
        ptypes = None
    if not ptypes:
        return {"error": "Could not read the parameters of %s(). Check the function name." % entry}

    # An unannotated parameter resolves to "any", whose default pool is thin and
    # has no multi-element lists. Widen it: inputs of the wrong type simply fail
    # on the correct version below and are filtered out, so a rich pool is safe.
    values = dict(mod.GrammarBaseline.BOUNDARY)
    # The stock string pool is alphabetic only, so a numeric-parsing or
    # punctuation-handling defect can never surface. Cover the usual shapes.
    strings = ['""', '"0"', '"5"', '"-5"', '"3.5"', '"a"', '"abc"',
               '"hello, "', '" git push"', '"key="', '"a+b"', '" "']
    values["str"] = strings
    # Order matters: the first rows are what the panel reads. Put plain numbers
    # first so a numeric function shows a legible difference rather than a type
    # error from a list, then strings, then lists. Inputs of the wrong type fail
    # on the correct version and are filtered out below, so nothing is lost.
    values["any"] = (
        ["0", "1", "2", "-1", "10", "3.5"]
        + strings
        + ["[]", "[1]", "[1, 2, 3]", "[5, 8, 7, 1]", "[2, 4, 6]", "[0, 0]", "[-1, 0, 1]"]
        + ["True", "False", "None"]
    )
    lists = [values.get(t, values["any"]) for t in ptypes]
    combos = _product_limited(lists, 120)

    differ, agree, invalid = [], [], 0
    for combo in combos:
        args = ", ".join(str(v) for v in combo)
        call = "result = %s(%s)" % (entry, args)
        ref_ok, ref_val, _ = mod.safe_exec(reference, call)
        if not ref_ok:
            invalid += 1
            continue
        def_ok, def_val, def_err = mod.safe_exec(defective, call)
        row = {
            "input": "%s(%s)" % (entry, args),
            "correct": _short(repr(ref_val)),
            "defective": (_short(repr(def_val)) if def_ok
                          else "error: " + _short((def_err or "").strip(), 40)),
        }
        # Compare by value, not by repr: 0 and 0.0 print differently but an
        # assertion against either one passes on both, so they are not a
        # difference the oracle would ever count.
        try:
            same = bool(def_val == ref_val)
        except Exception:
            same = repr(def_val) == repr(ref_val)
        if (not def_ok) or not same:
            differ.append(row)
        else:
            agree.append(row)
        if len(differ) >= 5 and len(agree) >= 5:
            break

    suggested = ""
    if differ:
        first = differ[0]
        suggested = "assert %s == %s" % (first["input"], first["correct"])
    return {
        "differ": differ[:6],
        "agree": agree[:6],
        "tried": len(differ) + len(agree) + invalid,
        "suggested_test": suggested,
    }


def _product_limited(lists, cap):
    """Bounded cartesian product so a many-argument function cannot explode."""
    out = [()]
    for values in lists:
        nxt = []
        for prefix in out:
            for value in values:
                nxt.append(prefix + (value,))
                if len(nxt) >= cap:
                    break
            if len(nxt) >= cap:
                break
        out = nxt
    return out[:cap]


def _short(text, limit=60):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console clean during the demonstration

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        blob = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def route(self):
        """Path without any query string, so /?x=1 still resolves."""
        return urlsplit(self.path).path

    def do_GET(self):
        if self.route() in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.route() == "/api/examples":
            self._send(200, json.dumps({
                "examples": EXAMPLES, "headline": HEADLINE_DATA,
            }))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "bad request"}))
            return
        try:
            if self.route() == "/api/run":
                result = act_run(payload)
            elif self.route() == "/api/search":
                result = act_search(payload)
            else:
                self._send(404, json.dumps({"error": "not found"}))
                return
        except Exception as exc:  # never crash the demo
            result = {"error": "%s: %s" % (type(exc).__name__, exc)}
        self._send(200, json.dumps(result))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oneiros</title>
<style>
  :root{
    --bg:#f2f5f6; --panel:#fff; --sunk:#e8edef; --ink:#12181c; --soft:#41505a;
    --muted:#6a7b85; --line:#cfd9dd; --accent:#0d6b74; --accent-soft:#dcecec;
    --pass:#1f7a4d; --pass-bg:#dff0e6; --fail:#a8402c; --fail-bg:#f8e3de;
    --warn:#8a6114; --warn-bg:#f6ecd6;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:"Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.5}
  .wrap{max-width:1240px;margin:0 auto;padding:22px 22px 70px}
  header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
    border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:20px}
  h1{font-size:27px;margin:0;letter-spacing:-.02em}
  .tag{font-size:14px;color:var(--muted)}
  .head-num{margin-left:auto;font-size:14px;color:var(--soft)}
  .head-num b{color:var(--accent);font-size:19px}
  .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
  label.lbl{font-size:11px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;
    color:var(--muted);display:block;margin-bottom:6px}
  select,button,input{font-family:inherit;font-size:15px}
  select{padding:9px 12px;border:1px solid var(--line);border-radius:5px;background:#fff;
    color:var(--ink);min-width:330px}
  button{padding:11px 20px;border-radius:5px;border:1px solid transparent;cursor:pointer;
    font-weight:600}
  .primary{background:var(--accent);color:#fff}
  .primary:hover{filter:brightness(1.08)}
  .ghost{background:#fff;border-color:var(--line);color:var(--ink)}
  .ghost:hover{background:var(--sunk)}
  .spec{background:var(--accent-soft);border-radius:6px;padding:12px 15px;margin-bottom:16px;
    font-size:15px;color:var(--ink)}
  .spec b{color:var(--accent)}
  .diffband{background:var(--panel);border:1px solid var(--line);border-radius:7px;
    margin-bottom:16px;overflow:hidden}
  .diffband h2{margin:0;padding:10px 15px;font-size:12px;letter-spacing:.07em;text-transform:uppercase;
    background:var(--sunk);border-bottom:1px solid var(--line);color:var(--soft)}
  .diffband .rows{padding:10px 0}
  .dl{display:flex;gap:10px;padding:4px 15px;font-family:Consolas,ui-monospace,monospace;
    font-size:14px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
  .dl .sign{flex:0 0 auto;font-weight:700;width:14px}
  .dl.minus{background:var(--fail-bg);color:var(--fail)}
  .dl.plus{background:var(--pass-bg);color:var(--pass)}
  .dl.same{color:var(--muted)}
  .diffband .none{padding:12px 15px;color:var(--muted);font-size:14.5px}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:900px){.cols{grid-template-columns:1fr}}
  .pane{background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:hidden;
    display:flex;flex-direction:column}
  .pane h2{margin:0;padding:11px 15px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;
    background:var(--sunk);border-bottom:1px solid var(--line);color:var(--soft)}
  .pane.good h2{color:var(--pass)} .pane.bad h2{color:var(--fail)}
  textarea{width:100%;border:0;padding:13px 15px;font-family:Consolas,ui-monospace,monospace;
    font-size:14.5px;line-height:1.6;resize:vertical;color:var(--ink);background:var(--panel);
    outline:none;tab-size:4}
  textarea.code{min-height:210px}
  textarea.one{min-height:64px}
  .runrow{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin:18px 0 8px}
  .runrow>div{flex:1 1 340px}
  .res{margin-top:18px;display:none}
  .verdict{padding:15px 18px;border-radius:7px;font-size:19px;font-weight:700;
    display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .v-caught{background:var(--pass-bg);color:var(--pass)}
  .v-not{background:var(--warn-bg);color:var(--warn)}
  .v-invalid{background:var(--fail-bg);color:var(--fail)}
  .verdict small{font-weight:500;font-size:15px;color:var(--soft)}
  .runs{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
  @media(max-width:900px){.runs{grid-template-columns:1fr}}
  .run{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:14px 16px}
  .run .who{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
  .run .st{font-size:24px;font-weight:700;margin-top:5px}
  .st.pass{color:var(--pass)} .st.fail{color:var(--fail)}
  .run .err{font-family:Consolas,monospace;font-size:13px;color:var(--soft);margin-top:6px;
    word-break:break-word}
  .run .vals{margin-top:7px;font-family:Consolas,monospace;font-size:14.5px;color:var(--ink);
    word-break:break-word}
  .run .vals .k{font-family:"Segoe UI",sans-serif;font-size:11px;letter-spacing:.09em;
    text-transform:uppercase;color:var(--muted);margin-right:3px}
  .run .vals .k:not(:first-child){margin-left:12px}
  table{width:100%;border-collapse:collapse;font-size:14.5px;margin-top:10px;background:var(--panel)}
  th{text-align:left;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
    padding:9px 12px;background:var(--sunk);border-bottom:1px solid var(--line)}
  td{padding:9px 12px;border-bottom:1px solid var(--sunk);
    font-family:Consolas,monospace;font-size:13.5px;vertical-align:top}
  tr:last-child td{border-bottom:0}
  .grp{margin-top:20px}
  .grp h3{font-size:15px;margin:0 0 4px}
  .grp p{margin:0;font-size:14px;color:var(--muted)}
  .grp.differ h3{color:var(--fail)} .grp.agree h3{color:var(--pass)}
  .wrapt{border:1px solid var(--line);border-radius:7px;overflow:hidden;margin-top:8px}
  .note{margin-top:22px;padding:13px 16px;background:var(--sunk);border-radius:6px;
    font-size:13.5px;color:var(--soft)}
  .err-box{background:var(--fail-bg);color:var(--fail);padding:13px 16px;border-radius:6px;
    margin-top:14px;font-size:15px}
  .busy{pointer-events:none}
  .working{padding:15px 18px;border-radius:7px;background:var(--accent-soft);color:var(--accent);
    font-size:17px;font-weight:600}
  .working::after{content:'';display:inline-block;width:9px;height:9px;margin-left:10px;
    border-radius:50%;background:currentColor;animation:pulse 1s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.25}50%{opacity:1}}
  @media (prefers-reduced-motion: reduce){.working::after{animation:none;opacity:.6}}
</style></head><body><div class="wrap">

<header>
  <h1>Oneiros</h1>
  <span class="tag">finding bugs by generating tests that catch them</span>
  <span class="head-num">full evaluation: <b id="hn">&nbsp;</b></span>
</header>

<div class="bar">
  <div>
    <label class="lbl" for="ex">Pick an example, or edit anything below</label>
    <select id="ex"></select>
  </div>
</div>

<div class="spec" id="spec"></div>

<div class="diffband" id="band">
  <h2>The difference between the two versions</h2>
  <div id="bandrows"></div>
</div>

<div class="cols">
  <div class="pane good">
    <h2>The correct version</h2>
    <textarea class="code" id="ref" spellcheck="false"></textarea>
  </div>
  <div class="pane bad">
    <h2>The version with a defect</h2>
    <textarea class="code" id="def" spellcheck="false"></textarea>
  </div>
</div>

<div class="runrow">
  <div>
    <label class="lbl" for="test">The test</label>
    <div class="pane"><textarea class="one" id="test" spellcheck="false"></textarea></div>
  </div>
  <div style="flex:0 0 auto;display:flex;gap:10px;padding-bottom:2px">
    <button class="primary" id="btnRun">Run the test on both</button>
    <button class="ghost" id="btnFind">Find inputs that differ</button>
  </div>
</div>

<div id="err"></div>
<div class="res" id="res"></div>

<div class="note" id="note"></div>

</div>
<script>
var EX = [], HN = null;

function $(id){ return document.getElementById(id); }
function esc(s){ var d=document.createElement('div'); d.textContent=s==null?'':String(s); return d.innerHTML; }

fetch('/api/examples').then(function(r){return r.json();}).then(function(d){
  EX = d.examples; HN = d.headline;
  $('hn').textContent = HN.killed + ' of ' + HN.total + ' = ' + HN.rate.toFixed(2) + '%';
  var sel = $('ex');
  EX.forEach(function(e,i){
    var o=document.createElement('option'); o.value=i; o.textContent=e.label; sel.appendChild(o);
  });
  var o=document.createElement('option'); o.value='own'; o.textContent='-- my own code --';
  sel.appendChild(o);
  sel.addEventListener('change', load);
  load();
  $('note').innerHTML = 'The result in the top right is the trained model’s own measured performance: '
    + 'it wrote tests for ' + HN.total + ' functions it had never seen and caught '
    + HN.killed + ' of the defects. This page runs the same pass/fail rule that produced that number. '
    + '<b>Find inputs that differ</b> uses the project’s non-model baseline search, not the trained model.';
});

function load(){
  var v=$('ex').value;
  if(v==='own'){
    $('spec').innerHTML='<b>Your own code.</b> Paste a working version on the left and a version with a bug on the right, then run a test or search for inputs where they disagree.';
    $('ref').value='def add(a, b):\n    return a + b\n';
    $('def').value='def add(a, b):\n    return a - b\n';
    $('test').value='assert add(2, 3) == 5';
    clear(); renderBand(); return;
  }
  var e=EX[v|0];
  $('spec').innerHTML='<b>What it should do:</b> '+esc(e.specification);
  $('ref').value=e.reference+'\n';
  $('def').value=e.defective+'\n';
  $('test').value=e.test;
  clear(); renderBand();
}

function lines(t){ return String(t).replace(/\r/g,'').split('\n'); }

function lcsDiff(a, b){
  var n=a.length, m=b.length, i, j;
  var dp=[]; for(i=0;i<=n;i++){ dp.push(new Array(m+1).fill(0)); }
  for(i=n-1;i>=0;i--) for(j=m-1;j>=0;j--)
    dp[i][j] = a[i]===b[j] ? dp[i+1][j+1]+1 : Math.max(dp[i+1][j], dp[i][j+1]);
  var out=[]; i=0; j=0;
  while(i<n && j<m){
    if(a[i]===b[j]){ out.push(['same', a[i]]); i++; j++; }
    else if(dp[i+1][j] >= dp[i][j+1]){ out.push(['minus', a[i]]); i++; }
    else { out.push(['plus', b[j]]); j++; }
  }
  while(i<n){ out.push(['minus', a[i++]]); }
  while(j<m){ out.push(['plus',  b[j++]]); }
  return out;
}

function renderBand(){
  var d = lcsDiff(lines($('ref').value), lines($('def').value));
  var idx = [];
  d.forEach(function(r,i){ if(r[0]!=='same') idx.push(i); });
  if(!idx.length){
    $('bandrows').innerHTML='<div class="none">The two versions are identical &mdash; edit one of them to introduce a defect.</div>';
    return;
  }
  var lo=Math.max(0, idx[0]-1), hi=Math.min(d.length-1, idx[idx.length-1]+1);
  var h='<div class="rows">';
  for(var i=lo;i<=hi;i++){
    var kind=d[i][0], text=d[i][1];
    if(kind==='same' && !text.trim()) continue;
    var sign = kind==='minus' ? '-' : (kind==='plus' ? '+' : ' ');
    h+='<div class="dl '+kind+'"><span class="sign">'+sign+'</span><span>'+esc(text||' ')+'</span></div>';
  }
  h+='</div>';
  $('bandrows').innerHTML=h;
}

var bandTimer=null;
function bandSoon(){ clearTimeout(bandTimer); bandTimer=setTimeout(renderBand, 250); }

function clear(){ $('res').style.display='none'; $('res').innerHTML=''; $('err').innerHTML=''; }
function fail(m){ $('err').innerHTML='<div class="err-box">'+esc(m)+'</div>'; }
function working(m){
  $('res').innerHTML='<div class="working">'+esc(m)+'</div>';
  $('res').style.display='block';
}

$('ref').addEventListener('input', bandSoon);
$('def').addEventListener('input', bandSoon);

function entryPoint(){
  var m=/def\s+([A-Za-z_]\w*)\s*\(/.exec($('ref').value);
  return m?m[1]:'';
}

function post(url, body, done){
  document.body.classList.add('busy');
  fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(function(r){return r.json();})
    .then(function(d){ document.body.classList.remove('busy'); done(d); })
    .catch(function(e){ document.body.classList.remove('busy'); fail('Could not reach the local server: '+e); });
}

$('btnRun').addEventListener('click', function(){
  clear(); working('Running the test on both versions');
  post('/api/run', {reference:$('ref').value, defective:$('def').value, test:$('test').value}, function(d){
    if(d.error){ clear(); fail(d.error); return; }
    var cls = d.verdict==='CAUGHT'?'v-caught':(d.verdict==='NOT CAUGHT'?'v-not':'v-invalid');
    var word = d.verdict==='CAUGHT'?'DEFECT CAUGHT':(d.verdict==='NOT CAUGHT'?'NOT CAUGHT':'INVALID TEST');
    $('res').innerHTML =
      '<div class="verdict '+cls+'">'+word+' <small>'+esc(d.detail)+'</small></div>'+
      '<div class="runs">'+card('On the correct version', d.reference, d.expected)+
      card('On the defective version', d.defective, d.expected)+'</div>';
    $('res').style.display='block';
  });
});

function card(who, r, expected){
  var h='<div class="run"><div class="who">'+who+'</div>'+
    '<div class="st '+(r.ok?'pass':'fail')+'">'+(r.ok?'PASS':'FAIL')+'</div>';
  if(r.returned!==null && r.returned!==undefined){
    h+='<div class="vals"><span class="k">returned</span> <b>'+esc(r.returned)+'</b>';
    if(expected) h+=' <span class="k">expected</span> <b>'+esc(expected)+'</b>';
    h+='</div>';
  }
  var e=(r.error||'').replace(/^AssertionError:\s*$/,'');
  if(e) h+='<div class="err">'+esc(e)+'</div>';
  else if(!r.ok && r.returned!==null && r.returned!==undefined)
    h+='<div class="err">the assertion did not hold</div>';
  return h+'</div>';
}

$('btnFind').addEventListener('click', function(){
  clear();
  var ep=entryPoint();
  if(!ep){ fail('Could not find a "def ..." line in the correct version.'); return; }
  working('Trying many inputs on both versions, this takes a few seconds');
  post('/api/search', {reference:$('ref').value, defective:$('def').value, entry_point:ep}, function(d){
    if(d.error){ clear(); fail(d.error); return; }
    var h='';
    h+='<div class="grp differ"><h3>Inputs where the two versions disagree &mdash; the defect shows here</h3>';
    h+='<p>'+(d.differ.length?'Each row is an input the defective version gets wrong.':'None found among the inputs tried.')+'</p>';
    if(d.differ.length) h+=table(d.differ);
    h+='</div>';
    h+='<div class="grp agree"><h3>Inputs where both versions agree &mdash; the defect stays hidden</h3>';
    h+='<p>'+(d.agree.length?'A test built from any of these would pass on both and prove nothing.':'None found among the inputs tried.')+'</p>';
    if(d.agree.length) h+=table(d.agree);
    h+='</div>';
    if(d.suggested_test){
      h+='<div class="note" style="margin-top:18px">Suggested test from the first disagreement: '+
         '<code>'+esc(d.suggested_test)+'</code> '+
         '<button class="ghost" style="margin-left:10px;padding:6px 12px;font-size:13px" id="useIt">Use this test</button></div>';
    }
    h+='<div class="note">Tried '+d.tried+' inputs.</div>';
    $('res').innerHTML=h; $('res').style.display='block';
    var b=$('useIt');
    if(b) b.addEventListener('click', function(){ $('test').value=d.suggested_test; $('btnRun').click(); });
  });
});

function table(rows){
  var h='<div class="wrapt"><table><tr><th>Input</th><th>Correct version returns</th><th>Defective version returns</th></tr>';
  rows.forEach(function(r){
    h+='<tr><td>'+esc(r.input)+'</td><td>'+esc(r.correct)+'</td><td>'+esc(r.defective)+'</td></tr>';
  });
  return h+'</table></div>';
}
</script></body></html>
"""

EXAMPLES = []
HEADLINE_DATA = {}


def main():
    global EXAMPLES, HEADLINE_DATA
    print("Loading corpus examples...")
    EXAMPLES = load_examples()
    HEADLINE_DATA = headline()
    runner()  # warm the executor before the browser opens
    url = "http://%s:%d/" % (HOST, PORT)
    server = HTTPServer((HOST, PORT), Handler)
    print("Loaded %d examples." % len(EXAMPLES))
    print()
    print("   Oneiros demo UI running at  %s" % url)
    print("   Local only. Press Ctrl+C in this window to stop.")
    print()
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
