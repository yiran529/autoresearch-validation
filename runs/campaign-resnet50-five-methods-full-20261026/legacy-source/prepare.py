"""ResNet50 full training; reuse frozen ResNet adapters and change common architecture."""
from copy import deepcopy
from pathlib import Path
import ast
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'campaigns/resnet18-five-methods-full-20261026'
METHODS=['greedylore-engineer-streamfix-resnet-v1','n023-engineer-maxu-resnet-v1',
         'n031-engineer-maxu-resnet-v1','n020-engineer-maxu-resnet-v1','dense']

def patch_driver(runtime):
    path=runtime/'pytorch-cifar/train_ddp.py';before=file_hash(path);text=path.read_text()
    substitutions={
        'from models.resnet import BasicBlock, ResNet':'from models.resnet import Bottleneck, ResNet',
        'ResNet(BasicBlock, [2, 2, 2, 2],':'ResNet(Bottleneck, [3, 4, 6, 3],',
        "model='ResNet18'":"model='ResNet50'",
        'name=output.name':"name=os.environ.get('WANDB_NAME', output.name)",
    }
    for old,new in substitutions.items():
        assert text.count(old)==1,(old,text.count(old))
        text=text.replace(old,new)
    anchor='    tracker = None\n';assert text.count(anchor)==1
    metadata="""    from glval.common import read_json as _read_method_json
    from glval.bridge import _observation as _hook_observation
    _method_record = _read_method_json(Path('method.json'))
    config.update(method_id=_method_record['id'], source_commit=_method_record['source_commit'],
                  method_parameters=_method_record['parameters'],
                  effective_hook_configuration=_hook_observation['record'].get('effective_state_configuration', {}),
                  resnet_block_type='Bottleneck', resnet_blocks=[3,4,6,3],
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
"""
    text=text.replace(anchor,metadata+anchor);ast.parse(text);path.write_text(text)
    return {'file':'pytorch-cifar/train_ddp.py','before':before,'after':file_hash(path),
        'architecture':'CommonResNet50: Bottleneck[3,4,6,3], existing CIFAR stem and classifier.',
        'other_changes':'Tracker name and actual method/precision metadata only; model/data/optimizer schedule common to all methods.'}

def main():
    recipe=read_json(ROOT/'recipes/resnet18-cifar10-muon-full-v1.json')
    recipe.update(id='resnet50-cifar10-five-methods-full-20261026',model_architecture='ResNet50',
        tracking={'mode':'online','project':'GreedyLore-CIFAR','group':'resnet50-five-methods-20261026'},
        checkpoint_policy='none',notes='CIFAR ResNet50 Bottleneck[3,4,6,3]; same full40epoch Muon recipe, selected method parameters and data as ResNet18.')
    cases=[]
    for phase in ['synthetic_check','full']:
        for method in METHODS:
            specific=deepcopy(recipe)
            if phase=='synthetic_check':
                specific['compression_start_step']=2;specific['args']['epochs']=205
                specific['completion']={'status':'smoke_completed','epoch':205,'update_step':205,'test_samples':128}
                specific['tracking']['mode']='disabled'
            name=phase+'-'+method;directory=HERE/'cases'/name;runtime=directory/'runtime'
            driver,metadata=build_runtime(runtime,ROOT/'methods'/method/'method.json',specific)
            change=patch_driver(runtime)
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
                'runtime':str(runtime),'output':str(directory/'output'),'driver':driver,'args':cli,'recipe':specific,
                'seed':1243,'expected_model':'ResNet50','expected_parameter_count':23520842,
                'runtime_files':inventory(runtime),'driver_patch':change,
                'scope':'Synthetic205step ResNet50 adapter check, compression2, refresh203;not full real-data validation' if phase=='synthetic_check' else 'RealCIFAR10 ResNet50 full40epochs/15640updates, no checkpoint resume'}
            write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'cases':cases,'methods':METHODS,'gpu_count':4,'full_recipe':recipe,
        'full_order':METHODS,'seed':1243,'repeats':1})
    runner=(PRIOR/'run.py').read_text()
    runner=runner.replace("WANDB_NAME=case['method_id']+'-s1243'","WANDB_NAME='resnet50-'+case['method_id']+'-s1243'")
    anchor="            if validation['status']=='verified':\n"
    check="""            config_path=output/'config.json'
            if config_path.exists():
                actual=read_json(config_path)
                if actual.get('model')!='ResNet50' or actual.get('parameter_count')!=23520842 or actual.get('resnet_blocks')!=[3,4,6,3] or actual.get('resnet_block_type')!='Bottleneck':
                    validation['status']='invalid';validation['errors'].append('ResNet50 architecture/parameter identity mismatch')
            else:
                validation['status']='invalid';validation['errors'].append('Missing model config')
"""
    assert runner.count(anchor)==1
    (HERE/'run.py').write_text(runner.replace(anchor,check+anchor))
    report=(PRIOR/'report.py').read_text().replace('ResNet18','ResNet50').replace('resnet18-five-methods-full-20261026','resnet50-five-methods-full-20261026').replace('glval-resnet18-full-20261026','glval-resnet50-full-20261026')
    (HERE/'report.py').write_text(report)
    write_json(HERE/'status.json',{'status':'prepared','gpu_execution':False})
    print({'cases':len(cases),'model':'ResNet50','gpu_count':4,'full_epochs':40,'gpu_execution':False})

if __name__=='__main__':main()
