"""Complete GreedyLoRE coverage for every screened setting, reusing selected confirmation."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent

def main():
    screen=read_json(HERE/'cases.json')
    selected=read_json(HERE/'confirmation/selection.json')['setting']['id']
    target=HERE/'greedy-all';target.mkdir(exist_ok=False)
    settings=[s for s in screen['settings'] if s['id']!=selected]
    cases=[]
    for setting in settings:
        template=next(c for c in screen['cases'] if c['setting']['id']==setting['id'])
        case=deepcopy(template)
        case.update(id=setting['id']+'-greedy-all',variant='greedy_big',
                    method='greedylore-engineer-streamfix',bucket_mb=2048,measured=200,
                    scope='All-setting GreedyLoRE coverage; seed1243;warm21 measure200; fullSVD cycle; not quality validation.')
        directory=target/'cases'/case['id'];runtime=directory/'runtime'
        recipe=read_json(Path(template['runtime'])/'recipe.json')
        recipe['args'].update(measured_iterations=200,ddp_bucket_cap_mb=2048)
        build_runtime(runtime,ROOT/'methods/greedylore-engineer-streamfix/method.json',recipe)
        case.update(runtime=str(runtime),output=str(directory/'output'),runtime_files=inventory(runtime))
        write_json(directory/'case.json',case);cases.append(case)
    write_json(target/'cases.json',{'settings':settings,'cases':cases,'repeats':1,
        'selected_setting_reused_from':'../confirmation','selected_setting':selected})
    for name in ['worker.py','run.py']:(target/name).write_text((HERE/'confirmation'/name).read_text())
    analyzer=(HERE/'confirmation/analyze.py').read_text()
    analyzer=analyzer.replace("VARIANTS=['dense_default','dense_big','n23_big','n20_big','n31_big','greedy_big']","VARIANTS=['greedy_big']")
    analyzer=analyzer.replace('| Dense默认 | Dense大bucket | N23大bucket | N20大bucket | N31大bucket | GreedyLoRE大bucket |','| GreedyLoRE大bucket |')
    analyzer=analyzer.replace('|---|---:|---:|---:|---:|---:|---:|','|---|---:|')
    (target/'analyze.py').write_text(analyzer)
    confirmation=read_json(HERE/'confirmation/cases.json')['cases']
    selected_greedy=[c for c in confirmation if c['method']=='greedylore-engineer-streamfix']
    others=[c for c in confirmation if c['method']!='greedylore-engineer-streamfix']
    # Complete all eight GreedyLoRE settings first, then finish the selected setting's other methods.
    followups=cases+selected_greedy+others
    assert len(followups)==13 and len({c['output'] for c in followups})==13
    write_json(HERE/'followups.json',{'cases':followups,'order':'All8GreedyLoRE settings first, then5selected-setting matched controls.'})
    runner=(HERE/'run_retry.py').read_text().replace("    if not args.only:parser.error('Explicit --only required for archived startup retry')\n",'')
    runner=runner.replace("HERE/'cases.json'","HERE/'followups.json'")
    runner=runner.replace("HERE/'hardware-retry.json'","HERE/'hardware-followups.json'").replace("HERE/'status-retry.json'","HERE/'status-followups.json'")
    (HERE/'run_followups.py').write_text(runner)
    print({'cases':len(cases),'reused_selected_setting':selected,'total_greedy_settings':8,'followup_cases':13,'gpu_execution':False})

if __name__=='__main__':main()
