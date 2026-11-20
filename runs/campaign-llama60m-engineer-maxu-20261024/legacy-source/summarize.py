"""One-shot report refresh; no training, waiting or GPU work."""
import csv,json,statistics
from pathlib import Path
from glval.common import read_json,write_json
from glval.plans import verify_plan
from glval.results import report
HERE=Path(__file__).resolve().parent
config=read_json(HERE/"campaign.json")
rows=[]
for stage,paths in [("pretraining",[q["run"] for q in config["training_queues"]]),("timing",[config["timing_run"]])]:
    for name in paths:
        run=Path(name);plan=verify_plan(run);result=report(run,plan)
        by_id={j["id"]:j for j in plan["jobs"]}
        for item in result["jobs"]:
            job=by_id[item["id"]];out=run/job["output"]
            tracking=read_json(out/"tracking.json") if (out/"tracking.json").exists() else {}
            rows.append({"stage":stage,"method":item["method_id"],"seed":item["seed"],"repetition":item["repetition"],
                         "status":item["status"],"wall_seconds":item["total_wall_seconds"],
                         "final_eval_loss":item["metrics"].get("final_eval_loss"),"final_eval_ppl":item["metrics"].get("final_eval_ppl"),
                         "mean_iteration_seconds":item["metrics"].get("mean_iteration_seconds"),"wandb_url":tracking.get("url"),"run":str(run)})
summary=[]
for method in dict.fromkeys(r["method"] for r in rows):
    samples=[r["mean_iteration_seconds"] for r in rows if r["stage"]=="timing" and r["method"]==method and r["status"]=="verified"]
    if samples:summary.append({"method":method,"repetitions_completed":len(samples),"mean_step_seconds":statistics.mean(samples),"median_step_seconds":statistics.median(samples),"relative_range":(max(samples)-min(samples))/statistics.mean(samples)})
write_json(HERE/"results.json",{"rows":rows,"timing_summary":summary,"limitations":["One pretraining seed. Full pretraining uses two GPU groups; total wall times across groups are not a controlled speed comparison.","Timing uses the same four GPUs after all pretraining finishes. Synthetic/short smoke is excluded."]})
with (HERE/"results.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({"results":str(HERE/"results.json"),"completed":sum(r["status"]=="verified" for r in rows),"total":len(rows)},indent=2))
