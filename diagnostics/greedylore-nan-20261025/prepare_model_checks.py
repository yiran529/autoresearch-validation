"""Build clearly marked short real-model diagnostics with the original LR schedule."""
import ast
from copy import deepcopy
from pathlib import Path
import shutil
from glval.common import ROOT,read_json,write_json,inventory
from glval.methods import create_method,check_method
from glval.build import build_runtime
from glval.plans import driver_args

HERE=Path(__file__).resolve().parent

class Instrument(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name!='main':return node
        body=[]
        for statement in node.body:
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id=='require_training_budget':
                body.append(statement)
                body.extend(ast.parse('diagnostic_finish(locals())').body)
                break
            body.append(self.visit(statement))
        node.body=body
        return node
    def visit_If(self,node):
        if 'args.eval_every' in ast.unparse(node.test):
            node.test=ast.Constant(False)
        return self.generic_visit(node)
    def visit_Assign(self,node):
        result=self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=='loss' for t in node.targets) and isinstance(node.value,ast.Attribute) and node.value.attr=='loss':
            return [result,*ast.parse('record_loss(loss, update_step + 1)').body]
        return result
    def visit_Expr(self,node):
        result=self.generic_visit(node)
        if ast.unparse(node)=='scaled_loss.backward()':
            return [result,*ast.parse('check_gradients(trainable_params, update_step + 1)').body]
        return result

def main():
    provenance=read_json(HERE/'source-provenance.json')
    mid='greedylore-engineer-streamfix'
    target=create_method(mid,HERE/'repo',provenance['fix_commit'])
    old=read_json(ROOT/'methods/greedylore-engineer/method.json')
    method=deepcopy(old)
    method.update(id=mid,source_commit=provenance['fix_commit'],selection_provenance={
        'parent':old['source_commit'],'derived_fix':provenance['fix_commit'],'U':None,
        'parameter_changes':False,'reason':'User-authorized correctness repair; not a searched/max-U candidate'})
    method['support_notes']['llama']='Original Engineer registration and parameters; explicit producer-to-NCCL callback dependency. Static review complete, diagnostic runtime validation pending.'
    method['reviewed_changes']['comm_hooks/subspace_hook.py']['reason']='Engineer source plus one producer CUDA event/wait in EF14; remove undefined SVD-output logging. No LR, rank, EF rule or optimizer change.'
    write_json(target/'method.json',method)
    (target/'adaptation.md').write_text('# GreedyLoRE Engineer stream dependency fix\n\n'
        f'Parent: `{old["source_commit"]}`\nFix: `{provenance["fix_commit"]}`\n\n'
        'Fix EF14 producer stream -> callback stream ordering using a CUDA event recorded after local R/EF production. '
        'The dense All-Reduce still overlaps local projection/reconstruction. Parameters, projection selection, EF update, optimizer and training loop are unchanged. '
        'SVD exception logging no longer reads undefined outputs. Original method and failed results retained.\n\n'
        'Synthetic four-rank delayed-producer test passed. Real-model/full-schedule validation recorded separately under diagnostics. '
        'No claim of full pretraining convergence from a short diagnostic.\n')
    check_method(target/'method.json','llama')
    cases=[('60m-old-early','greedylore-engineer','llama60m-adamw-full-online-v2',2,20),
           ('60m-fixed-early',mid,'llama60m-adamw-full-online-v2',2,20),
           ('60m-fixed-formal-transition',mid,'llama60m-adamw-full-online-v2',1000,1402),
           ('130m-fixed-formal-transition',mid,'llama130m-muon-full-online-v1',1000,1402)]
    manifest=[]
    for name,method_id,recipe_id,start,stop in cases:
        directory=HERE/'model-checks'/name
        runtime=directory/'runtime';output=directory/'output'
        recipe=read_json(ROOT/'recipes'/(recipe_id+'.json'))
        recipe.update(compression_start_step=start,tracking={'mode':'disabled','project':'GreedyLore-Validation','name':name})
        driver,method=build_runtime(runtime,ROOT/'methods'/method_id/'method.json',recipe)
        tree=Instrument().visit(ast.parse((runtime/driver).read_text()))
        tree.body.insert(0,ast.ImportFrom(module='diagnostic_guard',names=[ast.alias(name='record_loss'),
              ast.alias(name='check_gradients'),ast.alias(name='finish',asname='diagnostic_finish')],level=0))
        ast.fix_missing_locations(tree)
        (runtime/driver).write_text(ast.unparse(tree)+'\n')
        shutil.copyfile(HERE/'diagnostic_guard.py',runtime/'diagnostic_guard.py')
        args=driver_args(recipe,output,1243,method)+['--stop_after_steps',str(stop)]
        command=['/home/wyr/greedy_lore/.venv/bin/python','-B','-m','torch.distributed.run',
                 '--nnodes','1','--rdzv_backend','c10d','--rdzv_endpoint','127.0.0.1:0',
                 '--rdzv_id',name,'--nproc_per_node','4','--module','glval.worker','--driver',driver,'--',*args]
        record={'id':name,'runtime':str(runtime),'output':str(output),'command':command,
                'target_steps':stop,'original_scheduler_steps':recipe['args']['num_training_steps'],
                'compression_start':start,'runtime_files':inventory(runtime),
                'diagnostic_changes':['stop_after_steps','skip validation and final quality evaluation',
                                      'record finite forward loss and reduced-gradient norm each step'],
                'scope':'Not full pretraining; checkpoint saving disabled; W&B disabled; original training batch128 retained'}
        write_json(directory/'check.json',record)
        manifest.append(record)
    write_json(HERE/'model-checks.json',{'checks':manifest})
    print({'method':mid,'checks':[x['id'] for x in manifest],'gpu_execution':False})

if __name__=='__main__':main()
