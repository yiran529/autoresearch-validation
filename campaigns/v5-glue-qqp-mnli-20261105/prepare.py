"""Freeze additional tasks without changing live scaffold or existing runtimes."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import ast,json,sys,hashlib
from pathlib import Path
ROOT=Path("/home/wyr/greedylore-validation");sys.path.insert(0,str(ROOT))
from glval import plans
from glval.common import read_json,write_json
C=Path(__file__).parent
base=read_json(ROOT/"campaigns/v5-glue-sst2-qnli-lr2em5-20261104/qnli.json")
original_build=plans.build_runtime
def build_with_mnli_evaluation(target,method_path,recipe):
 driver,method=original_build(target,method_path,recipe)
 if recipe["args"]["task_name"]!="mnli":return driver,method
 p=target/driver;before=p.read_text();tree=ast.parse(before)
 class EvaluationBridge(ast.NodeTransformer):
  blocks=0;results=0
  def visit_If(self,node):
   if ast.unparse(node.test)=="args.task_name == 'mnli'":
    self.blocks+=1
    return ast.parse("""
if args.task_name == 'mnli':
    matched_eval_metric = dict(eval_metric)
    eval_dataset = processed_datasets['validation_mismatched']
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    model.eval()
    mismatched_evaluated_samples = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch['labels']))
        mismatched_evaluated_samples += len(references)
        metric.add_batch(predictions=predictions, references=references)
    assert mismatched_evaluated_samples == len(eval_dataset), 'MNLI mismatched sample coverage mismatch'
    eval_metric = metric.compute()
    logger.info(f'mnli-mm: {eval_metric}; samples={mismatched_evaluated_samples}')
""").body[0]
   return self.generic_visit(node)
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Name) and t.id=="all_results" for t in node.targets):
    self.results+=1
    return [node,*ast.parse("all_results.update(eval_matched_accuracy=matched_eval_metric['accuracy'], eval_mismatched_accuracy=eval_metric['accuracy'], eval_mismatched_samples=mismatched_evaluated_samples)").body]
   return self.generic_visit(node)
 bridge=EvaluationBridge();tree=ast.fix_missing_locations(bridge.visit(tree));assert bridge.blocks==bridge.results==1
 after=ast.unparse(tree)+"\n";compile(after,str(p),"exec");p.write_text(after)
 def training_loop(text):
  matches=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="epoch"]
  assert len(matches)==1
  return ast.dump(matches[0],include_attributes=False)
 assert training_loop(before)==training_loop(after)
 write_json(target/"mnli_evaluation_bridge.json",{"scope":"evaluation only; training loop AST unchanged","changes":["Preserve final matched accuracy separately from mismatched","Use gather_for_metrics to remove mismatched distributed tail padding","Disable autograd during mismatched evaluation; assert exact sample count"],"before_sha256":hashlib.sha256(before.encode()).hexdigest(),"after_sha256":hashlib.sha256(after.encode()).hexdigest(),"builder":str(Path(__file__).resolve())})
 return driver,method
plans.build_runtime=build_with_mnli_evaluation
if __name__ == "__main__":
 for task in ["qqp","mnli"]:
  exp=json.loads(json.dumps(base));recipe=read_json(ROOT/"recipes"/(base["recipe"]+".json"))
  recipe["id"]=f"glue-{task}-roberta-adamw-full-offline-lr2em5-v5";recipe["args"]["task_name"]=task
  recipe["metrics"]=["eval_accuracy","eval_f1"] if task=="qqp" else ["eval_matched_accuracy","eval_mismatched_accuracy"]
  recipe["notes"]+=" Additional queued task. QQP accuracy and F1; MNLI final matched/mismatched separately with evaluation-only padding correction. Shared LR2e-5/global16 across arms, no per-method tuning."
  write_json(ROOT/"recipes"/(recipe["id"]+".json"),recipe)
  exp["id"]=f"v5-glue-{task}-lr2em5-20261105";exp["recipe"]=recipe["id"];exp["execution"]["tracking"]["group"]=C.name
  write_json(C/(task+".json"),exp)
  run=plans.make_plan(C/(task+".json"));check=plans.preflight(run);assert not check["missing_inputs"]
  print(run,check["status"])
 
 