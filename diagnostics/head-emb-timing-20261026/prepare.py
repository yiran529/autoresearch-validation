"""GreedyLoRE head/embedding compression ablation; explicit, frozen variant."""
from copy import deepcopy
from pathlib import Path
import shutil
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime
HERE=Path(__file__).resolve().parent
PARENT='greedylore-engineer-streamfix'
VARIANT=PARENT+'-head-emb-v1'

def main():
    source=ROOT/'methods'/PARENT;target=ROOT/'methods'/VARIANT
    assert not target.exists()
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    method=read_json(target/'method.json');method.update(id=VARIANT)
    method['selection_provenance'].update(parent_method=PARENT,parent_method_hash=file_hash(source/'method.json'),
        change='User-requested inclusion of lm_head and embed_tokens in existing GreedyLoRE compression eligibility; no source or rank/EF change')
    method['support_notes']['llama']='Head/embedding inclusion experiment. Timing only; no convergence claim.'
    write_json(target/'method.json',method)
    adapter=(source/'adapter.py').read_text()
    old='''        return register_comm_hook_for_ddp_model(
            model, process_group, argparse.Namespace(**merged), optimizer=optimizer)'''
    new='''        from comm_hooks import utils
        original = utils.get_uncompress_names
        utils.get_uncompress_names = lambda ddp_model: []
        try:
            return register_comm_hook_for_ddp_model(
                model, process_group, argparse.Namespace(**merged), optimizer=optimizer)
        finally:
            utils.get_uncompress_names = original'''
    assert adapter.count(old)==1
    (target/'adapter.py').write_text(adapter.replace(old,new).replace('No algorithm rewrite.', 'Explicit head/embedding compression inclusion; native low-rank/EF unchanged.'))
    (target/'adaptation.md').write_text('User requested lm_head and embed_tokens compression. The adapter registers an empty name exclusion list. Native shape/rank eligibility and 1D exclusions remain. Parent source is unchanged; only compression coverage changes. Both independent embedding and output matrices remain separate parameters. Timing uses synthetic tokens and does not establish training quality.\n')
    assert inventory(source/'source')==inventory(target/'source')
    cases=[];settings=[]
    for model,dtype,length in [('130m','bfloat16',256),('130m','float32',64),('350m','bfloat16',256)]:
        setting={'id':f'{model}-{dtype}-s{length}-4gpu','model':model,'dtype':dtype,
            'sequence_length':length,'world_size':4,'batch_size':1,'optimizer':'adamw',
            'activation_checkpointing':False,'allow_tf32':False,'bucket_mb':2048}
        settings.append(setting)
        variants=[('dense','dense'),('greedy_exclude',PARENT),('greedy_include',VARIANT)]
        offset=(len(settings)-1)%3
        for label,method_id in variants[offset:]+variants[:offset]:
            recipe=read_json(ROOT/'recipes/llama130m-muon-timing-bsz1-local-v1.json')
            recipe.update(id='head-emb-'+setting['id'],world_size=4,compression_start_step=2,
                          tracking={'mode':'disabled','name':'head-emb-diagnostic'})
            recipe['args'].update(model_config=f'c4/configs/llama_{model}.json',optimizer='adamw',
                activation_checkpointing=False,batch_size=1,max_length=length,dtype=dtype,
                ddp_bucket_cap_mb=2048,warmup_iterations=21,measured_iterations=400,
                lr=.0025,beta1=.9,beta2=.999,eps=1e-8)
            recipe['completion'].update(measured_steps=400,warmup_iterations=21,first_measured_update=22,last_measured_update=421)
            name=setting['id']+'-'+label;directory=HERE/'cases'/name;runtime=directory/'runtime'
            build_runtime(runtime,ROOT/'methods'/method_id/'method.json',deepcopy(recipe))
            case={'id':name,'setting':setting,'variant':label,'model':model,'method':method_id,
                'runtime':str(runtime),'output':str(directory/'output'),'bucket_mb':2048,
                'local_muon':False,'no_ddp':False,'world_size':4,'batch_size':1,
                'sequence_length':length,'dtype':dtype,'allow_tf32':False,
                'profile':False,'profile_steps':0,'warmup':21,'measured':400,'seed':1243,
                'scope':'Synthetic cached tokens, actual model/AdamW updates; compression starts update3, measured updates22-421 cover refresh203/403. Single seed, four GPUs, no W&B/checkpoints. Performance only, not convergence.',
                'runtime_files':inventory(runtime)}
            write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'settings':settings,'cases':cases,'repeats':1,'parent':PARENT,'variant':VARIANT,
        'worker_hash':file_hash(HERE/'worker.py'),'runner_hash':file_hash(HERE/'run.py'),
        'notes':'All three arms freshly measured on identical GPUs, optimizer, bucket policy and 400-step refresh-inclusive window. No cross-hardware historical speed comparisons.'})
    write_json(HERE/'status.json',{'status':'prepared','gpu_execution':False})
    print({'cases':len(cases),'gpu_execution':False})
if __name__=='__main__':main()
