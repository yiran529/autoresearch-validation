"""ResNet50 full training; reuse frozen ResNet adapters and change common architecture."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

from copy import deepcopy
from pathlib import Path
import ast
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'campaigns/resnet18-five-methods-full-20261026'
METHODS=['greedylore-engineer-streamfix-resnet-v1', 'v5-n033-research-resnet-v1', 'dense']

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
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='parse_args':
            node.body=[item for item in node.body if not (
                isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name) and item.value.func.id=='add_muon_args'
                or isinstance(item,ast.If) and 'args.muon_scalar_' in ast.unparse(item.test))]
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='optimizer':
            node.value=ast.parse('torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=args.weight_decay)').body[0].value
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    text=text.replace('from optimizer import add_muon_args, build_muon_optimizer', '')
    text=text.replace('    args = parser.parse_args(argv)', "    parser.add_argument('--optimizer', choices=['adamw'], default='adamw')\n    args = parser.parse_args(argv)")
    text=text.replace("optimizer='muon'", "optimizer='adamw'").replace("orthogonalization_dtype='bfloat16'", 'orthogonalization_dtype=None')
    text=text.replace('DDP gradients before Muon momentum/orthogonalization','DDP gradients before AdamW update')
    text=text.replace("muon_collective_bits=optimizer.communication_bits_stats()['total']", 'optimizer_collective_bits=0')
    text=text.replace("muon_payload_scope='existing optimizer AllGather estimate: world_size*(world_size-1)*input_bits; different scope from per-rank DDP inputs, do not sum directly'", "optimizer_payload_scope='AdamW is local; no optimizer collectives'")
    anchor='    tracker = None\n';assert text.count(anchor)==1
    metadata="""    from glval.common import read_json as _read_method_json
    from glval.bridge import _observation as _hook_observation
    _method_record = _read_method_json(Path('method.json'))
    config.update(optimizer_class='torch.optim.AdamW', optimizer_betas=[0.9,0.999], optimizer_eps=1e-8, method_id=_method_record['id'], source_commit=_method_record['source_commit'],
                  method_parameters=_method_record['parameters'],
                  effective_hook_configuration=_hook_observation['record'].get('effective_state_configuration', {}),
                  resnet_block_type='Bottleneck', resnet_blocks=[3,4,6,3],
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
"""
    text=text.replace(anchor,metadata+anchor);ast.parse(text);path.write_text(text)
    return {'file':'pytorch-cifar/train_ddp.py','before':before,'after':file_hash(path),
        'architecture':'CommonResNet50: Bottleneck[3,4,6,3], existing CIFAR stem and classifier.',
        'optimizer':'torch.optim.AdamW; lr=0.005, betas=(0.9,0.999), eps=1e-8, decoupled weight_decay=0.0005',
        'other_changes':'Common AdamW replacement, tracker name and actual method/precision metadata only; model/data/optimizer schedule common to all methods.'}

def main():
    recipe=read_json(ROOT/'recipes/resnet18-cifar10-muon-full-v1.json')
    recipe.update(id='resnet50-cifar10-v5-n33-full-20261108',model_architecture='ResNet50',
        tracking={'mode':'online','project':'GreedyLore-CIFAR','group':'resnet50-adamw-v5-n33-20261108'},
        checkpoint_policy='none',notes='CIFAR ResNet50 Bottleneck[3,4,6,3]; full40epoch recipe with common AdamW lr0.005, betas0.9/0.999, eps1e-8, weight_decay0.0005, selected method parameters and data as ResNet18.')
    recipe['args']={k:v for k,v in recipe['args'].items() if not k.startswith('muon_')}
    recipe['args'].update(lr=0.005,optimizer='adamw')
    recipe['optimizer']={'class':'torch.optim.AdamW','betas':[0.9,0.999],'eps':1e-8,'decoupled_weight_decay':True}
    cases=[]
    for phase in ['full']:
        for method in METHODS:
            if phase == 'synthetic_check' and method != 'v5-n033-research-resnet-v1':
                continue
            specific=deepcopy(recipe)
            if phase=='synthetic_check':
                specific['compression_start_step']=2;specific['args']['epochs']=205
                specific['completion']={'status':'smoke_completed','epoch':205,'update_step':205,'test_samples':128}
                specific['tracking']['mode']='disabled'
            name=phase+'-'+method;directory=HERE/'cases'/name;runtime=directory/'runtime'
            driver,metadata=build_runtime(runtime,ROOT/'methods'/method/'method.json',specific)
            change=patch_driver(runtime)
            if method == 'v5-n033-research-resnet-v1':
                path = runtime / driver
                text = path.read_text().replace("uncompressed_names=['linear']", "uncompressed_names=[]")
                ast.parse(text)
                path.write_text(text)
                change['after'] = file_hash(path)
                change['compression_scope'] = 'Full DDP buckets including linear classifier'
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
    write_json(HERE/'status.json',{'status':'prepared','gpu_execution':False})
    print({'cases':len(cases),'model':'ResNet50','gpu_count':4,'full_epochs':40,'gpu_execution':False})

if __name__=='__main__':main()
