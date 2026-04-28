import json
from pathlib import Path

SEP = "=" * 70
BASE_DIR = Path(__file__).parent

for fname in ["results_baseline.json", "results_llm.json", "results_bm25.json",
              "results_dense.json", "results_hybrid.json", "results_crossencoder.json"]:
    print("\n" + SEP)
    print("FILE: " + fname)
    print(SEP)
    data = json.load(open(BASE_DIR / fname))
    for r in data:
        cfg = r.get("config", "?")
        ev = r.get("eval", {})
        score = ev.get("average_final_score", "ERROR")
        hard = {h["criteria_name"]: round(h["pass_rate"], 2) for h in ev.get("average_hard_scores", [])}
        soft = {s["criteria_name"]: round(s["average_score"], 2) for s in ev.get("average_soft_scores", [])}
        print("\n--- " + cfg + " ---")
        print("  avg_final_score : " + str(score))
        print("  hard pass rates : " + str(hard))
        print("  soft avg scores : " + str(soft))
        for ind in ev.get("individual_results", []):
            h_pass = all(h["passes"] for h in ind.get("hard_scores", []))
            fs = ind.get("final_score", 0)
            name = ind.get("candidate_name", "?")
            hard_detail = {h["criteria_name"]: h["passes"] for h in ind.get("hard_scores", [])}
            soft_detail = {s["criteria_name"]: s["score"] for s in ind.get("soft_scores", [])}
            status = "PASS" if h_pass else "FAIL"
            print("    [" + status + "] " + name.ljust(35) + " final=" + str(fs) + "  hard=" + str(hard_detail) + "  soft=" + str(soft_detail))
