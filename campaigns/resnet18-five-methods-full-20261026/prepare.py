"""Freeze full40epoch ResNet18 experiments and separately labelled synthetic checks."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

from copy import deepcopy
from pathlib import Path
import shutil
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent
PARENTS=['greedylore-engineer-streamfix','n023-engineer-maxu','n031-engineer-maxu','n020-engineer-maxu']

def patch_metadata(runtime):
    driver=runtime/'pytorch-cifar/train_ddp.py';before=file_hash(driver)
    text=driver.read_text()
    assert 'name=output.name' in text
    text=text.replace('name=output.name',"name=os.environ.get('WANDB_NAME', output.name)")
    # Driver already reports the recipe; add actual frozen method identity and
    # effective hook configuration so rank8 N20 is not labelled as rank32.
    anchor='    tracker = None\n'
    assert text.count(anchor)==1
    addition="""    from glval.common import read_json as _read_method_json
    from glval.bridge import _observation as _hook_observation
    _method_record = _read_method_json(Path('method.json'))
    config.update(method_id=_method_record['id'], source_commit=_method_record['source_commit'],
                  method_parameters=_method_record['parameters'],
                  effective_hook_configuration=_hook_observation['record'].get('effective_state_configuration', {}),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
"""
    driver.write_text(text.replace(anchor,addition+anchor))
    return {'file':'pytorch-cifar/train_ddp.py','before':before,'after':file_hash(driver),
            'reason':'Only tracker name and actual method/precision metadata; no model, update, data or budget change'}

def main():
    selected=[]
    for parent in PARENTS:
        original=ROOT/'methods'/parent;name=parent+'-resnet-v1';target=ROOT/'methods'/name
        assert not target.exists()
        shutil.copytree(original,target)
        method=read_json(target/'method.json')
        method.update(id=name,supported_workloads=['resnet'],support_notes={'resnet':
            'Native4D conv gradients flatten(out_channels,-1); canonical linear classifier excluded via thin registration bridge. Source unchanged; synthetic/real convergence evidence recorded separately.'})
        method['resnet_adaptation']={'parent_method':parent,'parent_method_hash':file_hash(original/'method.json'),
            'source_changed':False,'parameter_changes':False,'classifier_mapping':['linear']}
        for path,review in method['reviewed_changes'].items():
            if path.startswith('pytorch-cifar/'):
                review['reason']='Legacy candidate CIFAR entry not imported; canonical frozen train_ddp.py/models/optimizer provide shared ResNet recipe. Candidate communication hook overlay retained.'
        write_json(target/'method.json',method)
        (target/'adapter.py').write_text((HERE/'adapter.py').read_text())
        assert inventory(target/'source')==read_json(target/'source_manifest.json')['files']
        (target/'adaptation.md').write_text(f'''# {name}

Parent: {parent}; source commit {method['source_commit']}.
Source files and frozen key parameters identical to parent. No new search selection.
Candidate native4D convolution flattening retained. Dense1D/BatchNorm and linear
classifier use existing eligibility, with linear explicitly excluded as in the canonical driver.
Adapter temporarily supplies ResNet classifier names during public registration,
then restores the module helper even on exceptions. Muon update and data pipeline unchanged.
Static support declaration only; synthetic205step and full40epoch evidence live in this campaign.
Nominal rank32/gap200 retained; N20 internally caps effective rank8 and rejuvenates every10.
''')
        selected.append(name)
    selected.append('dense')
    recipe=read_json(ROOT/'recipes/resnet18-cifar10-muon-full-v1.json')
    recipe.update(id='resnet18-cifar10-five-methods-full-20261026',tracking={
        'mode':'online','project':'GreedyLore-CIFAR','group':'resnet18-five-methods-20261026'},checkpoint_policy='none')
    cases=[]
    for phase in ['synthetic_check','full']:
        for method in selected:
            specific=deepcopy(recipe)
            if phase=='synthetic_check':
                specific['compression_start_step']=2
                specific['args']['epochs']=205
                specific['completion']={'status':'smoke_completed','epoch':205,'update_step':205,'test_samples':128}
                specific['tracking']['mode']='disabled'
            name=phase+'-'+method;directory=HERE/'cases'/name;runtime=directory/'runtime'
            driver,metadata=build_runtime(runtime,ROOT/'methods'/method/'method.json',specific)
            patch=patch_metadata(runtime)
            args=dict(specific['args'])
            args.update({'output-dir':str(directory/'output'),'seed':1243,
                'compressor':metadata['parameters']['compressor'],
                'compress-rank':metadata['parameters'].get('compress_rank',32),
                'update-proj-gap':metadata['parameters'].get('update_proj_gap',200),
                'min-compression-rate':metadata['parameters'].get('min_compression_rate',1.),
                'start-compress-iter':specific['compression_start_step'],
                'wandb-mode':specific['tracking']['mode'],'wandb-project':specific['tracking']['project']})
            if phase=='synthetic_check':args.update({'smoke-steps':205,'workers':0})
            cli=[]
            for key,value in args.items():
                if value is False:continue
                cli.append('--'+key)
                if value is not True:cli.append(str(value))
            case={'id':name,'method_id':method,'phase':phase,'source_commit':metadata['source_commit'],
                  'runtime':str(runtime),'output':str(directory/'output'),'driver':driver,'args':cli,
                  'recipe':specific,'seed':1243,'runtime_files':inventory(runtime),'metadata_patch':patch,
                  'scope':'Synthetic205updates with compression2 and refresh203; not real-data validation' if phase=='synthetic_check' else 'RealCIFAR10 full40epochs/15640updates; no checkpoint resume'}
            write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'cases':cases,'methods':selected,'gpu_count':4,
        'full_recipe':recipe,'full_order':selected,'seed':1243,'repeats':1})
    write_json(HERE/'status.json',{'status':'prepared','gpu_execution':False})
    print({'cases':len(cases),'methods':selected,'full_epochs':40,'gpu_execution':False})

if __name__=='__main__':main()
