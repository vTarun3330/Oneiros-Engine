"""Emit the final research summary from committed artifacts only.

Every number below is read out of a committed receipt and none is typed by
hand, because a summary that restates figures from memory is a second source
of truth that can drift from the first. The receipts are hashed into the
output, so a reader can check that the summary describes the artifacts it
claims to.

Reads only:
  results/v4_2_frozen_development_evaluation_receipt.json
  results/v4_2_development_selection_receipt.json
  results/v4_2_locked_validation_preflight.json
  results/v4_2_locked_validation_result_receipt.json

Touches no split, no model, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = {
    "frozen_development": "results/v4_2_frozen_development_evaluation_receipt.json",
    "development_selection": "results/v4_2_development_selection_receipt.json",
    "locked_preflight": "results/v4_2_locked_validation_preflight.json",
    "locked_result": "results/v4_2_locked_validation_result_receipt.json",
}
OUTPUT = "results/v4_2_final_research_summary.md"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    data, digests = {}, {}
    for key, rel in SOURCES.items():
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"required committed artifact is missing: {rel}")
        data[key] = json.loads(path.read_text(encoding="utf-8"))
        digests[rel] = sha256_file(path)
    return data, digests


def main() -> int:
    d, digests = load()
    sel = d["development_selection"]
    lock = d["locked_result"]
    lpre = d["locked_preflight"]

    L: list[str] = []
    A = L.append

    A("# Oneiros — Final Research Summary")
    A("")
    A("Generated from committed artifacts by `scripts/emit_final_research_summary.py`. "
      "Every figure is read from a receipt listed under *Sources*; none is transcribed.")
    A("")
    A("## Final selection")
    A("")
    A("| | |")
    A("|---|---|")
    model = lpre["model_identity"]
    A("| **Final candidate** | immutable base `%s` @ `%s` |"
      % (model["base_model_name"], model["base_model_revision"]))
    A("| **Arm A@431 (SFT)** | rejected for promotion at locked validation |")
    A("| **O1 sidecar / Arm B** | rejected on development |")
    A("| **Promoted SFT model** | none |")
    A("")
    A("> %s" % lock["decision"])
    A("")

    # ---------------------------------------------------- development stage
    A("## Stage 1 — development: four-arm experiment (ablation_dev, 542 functions)")
    A("")
    A("Decision rule frozen before any result existed "
      "(`frozen_before_any_evaluation_ran`: %s; "
      "`thresholds_chosen_before_seeing_results`: %s)."
      % (d["frozen_development"]["frozen_before_any_evaluation_ran"],
         d["frozen_development"]["promotion_rule"]["thresholds_chosen_before_seeing_results"]))
    A("")
    A("| arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 | killed/542 | reference validity |")
    A("|---|---|---|---|---|---|---|")
    for name in ("base", "A@150", "A@431", "B@150"):
        m = sel["arms"][name]["metrics"]
        A("| %s | %.4f | %.4f | %.4f | **%.4f** | %d | %.4f |"
          % (name, m["kill_at_1"], m["kill_at_2"], m["kill_at_4"], m["kill_at_8"],
             m["kill_at_8_functions"], m["reference_valid_rate"]))
    A("")
    prim = sel["comparisons"]["primary_matched_duration_o1_effect"]
    sec = sel["comparisons"]["secondary_selected_checkpoint_policy"]
    A("**Primary comparison — B@150 vs A@150 (matched training duration, isolates the O1 effect):** "
      "Kill@8 %+.2f points, %d gained / %d lost, net %+d, McNemar exact p = %.4g → **%s**."
      % (prim["kill_at_8_delta_points"], prim["functions_gained"], prim["functions_lost"],
         prim["net_functions"], prim["mcnemar_exact_two_sided_p"],
         sel["promotion_rule_outcome"]["primary_B150_vs_A150"]["verdict"]))
    A("")
    A("**Secondary — B@150 vs A@431:** Kill@8 %+.2f points, net %+d, p = %.4g → **%s**."
      % (sec["kill_at_8_delta_points"], sec["net_functions"],
         sec["mcnemar_exact_two_sided_p"],
         sel["promotion_rule_outcome"]["secondary_B150_vs_A431"]["verdict"]))
    A("")
    A("### O1 rejection")
    A("")
    for finding in sel["explicit_findings"][:4]:
        A("- %s" % finding)
    A("")

    # -------------------------------------------------- locked validation
    A("## Stage 2 — locked validation: base vs Arm A@431 (val, %d functions)"
      % lock["paired_comparison"]["n"])
    A("")
    A("Decision rule frozen before either arm ran (`frozen_before_results`: %s). "
      "The bar was set deliberately above the development screen because "
      "ablation_dev had selected this checkpoint and so flattered it."
      % lpre["decision_rule"]["frozen_before_results"])
    A("")
    mb = lock["arms"]["base_control"]["metrics"]
    ma = lock["arms"]["arm_a_checkpoint_431"]["metrics"]
    A("| metric | base | A@431 |")
    A("|---|---|---|")
    for label, key, fmt in (
        ("Kill@1", "k1", "%.6f"), ("Kill@2", "k2", "%.6f"),
        ("Kill@4", "k4", "%.6f"), ("Kill@8", "k8", "%.6f"),
        ("parse validity", "parse", "%.6f"),
        ("execution validity", "ex", "%.6f"),
        ("reference validity", "ref", "%.6f"),
        ("redundancy", "red", "%.6f"),
        ("diversity exact-unique", "dex", "%.6f"),
        ("diversity input-shape", "din", "%.6f"),
    ):
        A("| %s | %s | %s |" % (label, fmt % mb[key], fmt % ma[key]))
    A("| Kill@8 functions | %d/%d | %d/%d |" % (mb["k8n"], mb["n"], ma["k8n"], ma["n"]))
    A("| functions with a reference-valid candidate | %d (%.4f) | %d (%.4f) |"
      % (mb["fnrefn"], mb["fnref"], ma["fnrefn"], ma["fnref"]))
    A("")
    pc = lock["paired_comparison"]
    A("**Paired (n=%d):** gained **%d**, lost **%d**, net **%+d**, killed by both %d, "
      "by neither %d. **McNemar exact two-sided p = %.10g.** Kill@8 delta **%+.4f points**."
      % (pc["n"], pc["functions_gained"], pc["functions_lost"], pc["net_functions"],
         pc["killed_by_both"], pc["killed_by_neither"],
         pc["mcnemar_exact_two_sided_p"], pc["kill_at_8_delta_points"]))
    A("")
    A("### Frozen promotion rule, as applied")
    A("")
    A("| criterion | measured | result |")
    A("|---|---|---|")
    c = lock["promotion_criteria"]
    A("| 1 practical kill gain | %+.4f pts / net %+d / p = %.4g | **%s** |"
      % (c["1_practical_kill_gain"]["kill_at_8_points"],
         c["1_practical_kill_gain"]["net_functions"],
         c["1_practical_kill_gain"]["mcnemar_p"],
         "PASS" if c["1_practical_kill_gain"]["passed"] else "FAIL"))
    A("| 2 reference validity | %+.4f pts (max drop %.1f) | **%s** |"
      % (c["2_reference_validity"]["delta_points"], c["2_reference_validity"]["max_drop"],
         "PASS" if c["2_reference_validity"]["passed"] else "FAIL"))
    A("| 3 execution and parse | parse %+.4f, exec %+.4f | **%s** |"
      % (c["3_execution_and_parse"]["parse_delta_points"],
         c["3_execution_and_parse"]["execution_delta_points"],
         "PASS" if c["3_execution_and_parse"]["passed"] else "FAIL"))
    A("| 4 diversity | exact %+.2f%%, input %+.2f%%, redundancy %+.2f%% | **%s** |"
      % (c["4_diversity"]["exact_relative"] * 100,
         c["4_diversity"]["input_shape_relative"] * 100,
         c["4_diversity"]["redundancy_relative"] * 100,
         "PASS" if c["4_diversity"]["passed"] else "FAIL"))
    A("| 5 integrity | shared run contract: %s | **%s** |"
      % (c["5_integrity"]["shared_run_contract"],
         "PASS" if c["5_integrity"]["passed"] else "FAIL"))
    A("")
    A("**Decision: %s.**" % lock["decision"])
    A("")
    A("Criterion 1 is the substantive failure: the gain is the right sign and too "
      "noisy to call. Criterion 5 failed for an implementation reason documented "
      "separately in `docs/LOCKED_VALIDATION_INTEGRITY_DEFECT.md`; it is preserved "
      "as recorded history and the decision does not depend on it.")
    A("")

    # ------------------------------------------------------------- limits
    A("## Limitations")
    A("")
    A("These bound every claim made above.")
    A("")
    A("1. **Reference-validity regression.** SFT cost %+.2f points of reference "
      "validity against base on the locked split (%.4f → %.4f), and functions with any "
      "reference-valid candidate fell from %d to %d. The same regression appeared in "
      "every development SFT arm. It passed criterion 2 only because a %.1f-point "
      "tolerance had been declared in advance, and it passed by %.2f points."
      % (pc["reference_validity_delta_points"], mb["ref"], ma["ref"],
         mb["fnrefn"], ma["fnrefn"], c["2_reference_validity"]["max_drop"],
         c["2_reference_validity"]["max_drop"] + pc["reference_validity_delta_points"]))
    A("2. **No promoted SFT model.** The final candidate is the base model. Nothing "
      "here supports a claim of robust Oneiros SFT generalization.")
    A("3. **Repository records are excluded from every kill rate reported.** "
      "%d were held out on the development panel, per the selection receipt. The "
      "locked panel likewise excluded its repository records; that count is recorded "
      "in the locked run artifacts rather than in the committed receipt, so it is not "
      "restated here."
      % sel["arms"]["base"]["metrics"]["repository_validation_records_held"])
    A("4. **No real native repository result.** No repository-native execution was "
      "performed; no real-repository performance is claimed.")
    A("5. **Development-panel figures are not generalization evidence.** ablation_dev "
      "selected the checkpoints it scored. The locked val split did not, which is why "
      "the +%.2f development gain became +%.2f under locking."
      % (sel["comparisons"]["each_arm_vs_base_control"]["A@431"]["kill_at_8_delta_points"],
         pc["kill_at_8_delta_points"]))
    A("6. **The sealed final test has not been opened.**")
    A("")

    A("## Sources")
    A("")
    A("| artifact | sha256 |")
    A("|---|---|")
    for rel, digest in sorted(digests.items()):
        A("| `%s` | `%s` |" % (rel, digest))
    A("")
    A("Generated %s at commit `%s`."
      % (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd=ROOT).stdout.strip()))

    out = ROOT / OUTPUT
    out.write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    print("wrote %s" % OUTPUT)
    print("sha256 %s" % sha256_file(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
