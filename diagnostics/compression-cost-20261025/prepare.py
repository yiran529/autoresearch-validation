"""Build isolated performance diagnostics. No quality or full-recipe claims."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent
NOOP='''import torch
def noop(state,bucket):
    future=torch.futures.Future()
    future.set_result(bucket.buffer())
    return future
class MethodAdapter:
    def validate(self,context,config):pass
    def describe(self):return {"diagnostic":"No gradient communication; not a valid distributed training method"}
    def attach(self,model,optimizer,process_group,args,context,config):
        model.register_comm_hook(None,noop)
'''

def main():
    cases=[]
    for model in ['60m','130m']:
        recipe=read_json(ROOT/'recipes'/('llama'+model+'-'+('adamw' if model=='60m' else 'muon')+'-timing-bsz1-local-v1.json'))
        recipe.update(compression_start_step=2,tracking={'mode':'disabled','name':'performance-diagnostic'})
        variants=[('dense','dense',25,False),('no_ddp','dense',25,False),
                  ('n23','n023-engineer-maxu',25,False),('n31','n031-engineer-maxu',25,False),
                  ('n20','n020-engineer-maxu',25,False),('greedy_fixed','greedylore-engineer-streamfix',25,False),
                  ('dense_big','dense',1024,False),('n23_big','n023-engineer-maxu',1024,False),
                  ('n31_big','n031-engineer-maxu',1024,False)]
        if model=='130m':variants += [('dense_local_muon','dense',25,True),('n31_local_muon','n031-engineer-maxu',25,True)]
        for suffix,method,bucket,local_muon in variants:
            name=model+'-'+suffix;directory=HERE/'cases'/name
            runtime=directory/'runtime'
            driver,_=build_runtime(runtime,ROOT/'methods'/method/'method.json',recipe)
            if suffix=='no_ddp':(runtime/'candidate_adapter.py').write_text(NOOP)
            case={'id':name,'model':model,'method':method,'runtime':str(runtime),
                  'output':str(directory/'output'),'bucket_mb':bucket,'local_muon':local_muon,
                  'no_ddp':suffix=='no_ddp','profile':bucket==25 and suffix not in ['no_ddp','dense_local_muon','n31_local_muon'],
                  'warmup':11,'measured':30,'profile_steps':3,
                  'scope':'Cached synthetic tokens; early compression; steady-state diagnostic, not original800-step timing/quality',
                  'runtime_files':inventory(runtime)}
            write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'cases':cases})
    print({'cases':len(cases),'gpu_execution':False})

if __name__=='__main__':main()
