"""Complete missing N31 rows and the former four-card Dense-only reference."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent
PRECISION=ROOT/'diagnostics/timing-precision-seq-20261025'
OLDER=ROOT/'diagnostics/timing-settings-20261025'

def main():
    base=read_json(PRECISION/'cases.json');cases=[];settings=[]
    selected=read_json(PRECISION/'confirmation/selection.json')['setting']['id']
    for setting in base['settings']:
        if setting['id']==selected:continue
        template=next(c for c in base['cases'] if c['setting']['id']==setting['id'] and c['variant']=='n23_big')
        case=deepcopy(template);case.update(id=setting['id']+'-n31-complete',variant='n31_big',method='n031-engineer-maxu')
        directory=HERE/'cases'/case['id'];runtime=directory/'runtime'
        recipe=read_json(Path(template['runtime'])/'recipe.json')
        build_runtime(runtime,ROOT/'methods/n031-engineer-maxu/method.json',recipe)
        case.update(runtime=str(runtime),output=str(directory/'output'),runtime_files=inventory(runtime))
        write_json(directory/'case.json',case);cases.append(case);settings.append(setting)
    old=next(c for c in read_json(OLDER/'cases.json')['cases'] if c['id']=='350m-adam-4-default-dense')
    setting={'id':'350m-adam-4-default-complete','model':'350m','dtype':'bfloat16','sequence_length':256,
             'world_size':4,'batch_size':1,'optimizer':'adamw','activation_checkpointing':False,'allow_tf32':False}
    settings.append(setting)
    for variant,method in [('dense_default','dense'),('n23_default','n023-engineer-maxu'),
                           ('n31_default','n031-engineer-maxu'),('n20_default','n020-engineer-maxu'),
                           ('greedy_default','greedylore-engineer-streamfix')]:
        case={'id':setting['id']+'-'+variant,'setting':setting,'variant':variant,'model':'350m','method':method,
              'bucket_mb':None,'local_muon':False,'no_ddp':False,'world_size':4,'batch_size':1,'sequence_length':256,
              'dtype':'bfloat16','allow_tf32':False,'profile':False,'profile_steps':0,'warmup':21,'measured':200,'seed':1243,
              'scope':'Complete former4GPU Dense-only reference;synthetic;seed1243;warm21 measure200;fullGreedy SVD cycle;not quality validation.'}
        directory=HERE/'cases'/case['id'];runtime=directory/'runtime'
        recipe=read_json(Path(old['runtime'])/'recipe.json')
        build_runtime(runtime,ROOT/'methods'/method/'method.json',recipe)
        case.update(runtime=str(runtime),output=str(directory/'output'),runtime_files=inventory(runtime))
        write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'settings':settings,'cases':cases,'repeats':1,'reused_n31_setting':selected})
    for name,source in [('worker.py',PRECISION/'worker.py'),('run.py',PRECISION/'confirmation/run.py')]:
        (HERE/name).write_text(source.read_text())
    analyzer=(PRECISION/'analyze.py').read_text()
    analyzer=analyzer.replace('len(ranks)==case[\'world_size\']==8',"len(ranks)==case['world_size']")
    analyzer=analyzer.replace('list(range(8))',"list(range(case['world_size']))")
    analyzer=analyzer.replace("result['global_batch']==8","result['global_batch']==case['world_size']*case['batch_size']")
    analyzer=analyzer.replace('[1]*8',"[1]*case['world_size']").replace('for i in range(8)',"for i in range(case['world_size'])")
    analyzer=analyzer.replace('8GPUs','4/8GPUs')
    analyzer=analyzer.replace("assert state['hook_registrations']==1", "assert state['hook_registrations']==1\n                    if case['method']=='greedylore-engineer-streamfix':\n                        assert state['iter']==221\n                        assert max(range(200),key=lambda i:r['steps'][i])+22==203")
    (HERE/'analyze.py').write_text(analyzer)
    print({'new_cases':len(cases),'n31_missing':7,'fresh_four_card_reference':5,'gpu_execution':False})

if __name__=='__main__':main()
