"""Select the strongest screened setting and compare all methods over one SVD cycle."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent

def main():
    analysis=read_json(HERE/'analysis.json')
    assert analysis['finished']==analysis['valid']==32
    eligible=[]
    for row in analysis['rows']:
        dense=min(row['variants'][v]['mean_step_ms'] for v in ['dense_default','dense_big'])
        compressed=min(row['variants'][v]['mean_step_ms'] for v in ['n23_big','n20_big'])
        eligible.append((dense/compressed,row))
    ratio,selected=max(eligible,key=lambda pair:pair[0])
    assert ratio>1.05,('No clear setting benefit to confirm',ratio)
    setting=deepcopy(selected['setting'])
    target=HERE/'confirmation';target.mkdir(exist_ok=False)
    cases=[]
    variants=[('dense_default','dense',None),('dense_big','dense',2048),
              ('n23_big','n023-engineer-maxu',2048),('n20_big','n020-engineer-maxu',2048),
              ('n31_big','n031-engineer-maxu',2048),('greedy_big','greedylore-engineer-streamfix',2048)]
    template=next(c for c in read_json(HERE/'cases.json')['cases'] if c['setting']['id']==setting['id'])
    for suffix,method,bucket in variants:
        case=deepcopy(template);case.update(id=setting['id']+'-confirm-'+suffix,variant=suffix,method=method,
                                          bucket_mb=bucket,measured=200,
                                          scope='Selected setting confirmation; seed1243; warm21 measure200; one fullGreedy SVD cycle; no quality claim.')
        directory=target/'cases'/case['id'];runtime=directory/'runtime'
        recipe=read_json(Path(template['runtime'])/'recipe.json')
        recipe['args'].update(measured_iterations=200,ddp_bucket_cap_mb=bucket)
        build_runtime(runtime,ROOT/'methods'/method/'method.json',recipe)
        case.update(runtime=str(runtime),output=str(directory/'output'),runtime_files=inventory(runtime))
        write_json(directory/'case.json',case);cases.append(case)
    write_json(target/'cases.json',{'settings':[setting],'cases':cases,'repeats':1})
    write_json(target/'selection.json',{'source_analysis_hash':file_hash(HERE/'analysis.json'),
        'criterion':'Largest screened bestDense/best(N23,N20) ratio; confirm independently over200 measured steps and addN31/Greedy.',
        'screened_speedup':ratio,'setting':setting,'seed':1243,
        'note':'Same seed across screen and confirmation, no multi-seed quality or statistical significance claim.'})
    (target/'worker.py').write_text((HERE/'worker.py').read_text())
    runner=(HERE/'run_retry.py').read_text().replace("    if not args.only:parser.error('Explicit --only required for archived startup retry')\n",'')
    runner=runner.replace("HERE/'hardware-retry.json'","HERE/'hardware.json'").replace("HERE/'status-retry.json'","HERE/'status.json'")
    (target/'run.py').write_text(runner)
    analyzer=(HERE/'analyze.py').read_text()
    analyzer=analyzer.replace("VARIANTS=['dense_default','dense_big','n23_big','n20_big']", "VARIANTS=['dense_default','dense_big','n23_big','n20_big','n31_big','greedy_big']")
    analyzer=analyzer.replace("assert state['hook_registrations']==1", "assert state['hook_registrations']==1\n                    if case['method']=='greedylore-engineer-streamfix':\n                        assert state['iter']==221\n                        assert max(range(200),key=lambda i:r['steps'][i])+22==203\n                        source=(Path(case['runtime'])/state['hook_source']).read_text()\n                        assert '((state.iter - state.start_compress_iter) % state.update_proj_gap == 0)' in source")
    analyzer=analyzer.replace('measure100','measure200').replace('测量100步','测量200步').replace('包含10个rejuvenation周期','包含20个rejuvenation周期；GreedyLoRE包含一次SVD刷新')
    analyzer=analyzer.replace('| N20大bucket |','| N20大bucket | N31大bucket | GreedyLoRE大bucket |').replace('|---|---:|---:|---:|---:|','|---|---:|---:|---:|---:|---:|---:|')
    analyzer=analyzer.replace("['prepare.py','worker.py','run.py','analyze.py']", "['worker.py','run.py','analyze.py']")
    (target/'analyze.py').write_text(analyzer)
    print({'setting':setting['id'],'screened_speedup':ratio,'cases':len(cases),'gpu_execution':False})

if __name__=='__main__':main()
