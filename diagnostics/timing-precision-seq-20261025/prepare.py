"""Eight-GPU dtype/sequence factorial screen; all writes stay in the scaffold."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent

def main():
    settings=[];cases=[]
    for model in ['130m','350m']:
        for dtype,length in [('bfloat16',256),('bfloat16',64),('float32',64),('float32',256)]:
            precision='bf16' if dtype=='bfloat16' else 'fp32'
            setting={'id':f'{model}-{precision}-s{length}','model':model,'dtype':dtype,
                     'sequence_length':length,'world_size':8,'batch_size':1,
                     'optimizer':'adamw','activation_checkpointing':False,'allow_tf32':False}
            settings.append(setting)
            variants=[('dense_default','dense',None),('dense_big','dense',2048),
                      ('n23_big','n023-engineer-maxu',2048),('n20_big','n020-engineer-maxu',2048)]
            offset=(len(settings)-1)%len(variants)
            for suffix,method,bucket in variants[offset:]+variants[:offset]:
                recipe=read_json(ROOT/'recipes/llama130m-muon-timing-bsz1-local-v1.json')
                recipe.update(world_size=8,compression_start_step=2,
                              tracking={'mode':'disabled','name':'precision-sequence-diagnostic'})
                recipe['args'].update(model_config=f'c4/configs/llama_{model}.json',
                    optimizer='adamw',activation_checkpointing=False,batch_size=1,max_length=length,
                    dtype=dtype,ddp_bucket_cap_mb=bucket,warmup_iterations=21,measured_iterations=100,
                    lr=.0025,beta1=.9,beta2=.999,eps=1e-8)
                name=setting['id']+'-'+suffix;directory=HERE/'cases'/name;runtime=directory/'runtime'
                build_runtime(runtime,ROOT/'methods'/method/'method.json',deepcopy(recipe))
                case={'id':name,'setting':setting,'variant':suffix,'model':model,'method':method,
                      'runtime':str(runtime),'output':str(directory/'output'),
                      'bucket_mb':bucket,'local_muon':False,'no_ddp':False,'world_size':8,
                      'batch_size':1,'sequence_length':length,'dtype':dtype,'allow_tf32':False,
                      'profile':False,'profile_steps':0,'warmup':21,'measured':100,'seed':1243,
                      'scope':'Cached synthetic token prefix; compression starts2; updates22-121. N20 covers10 rejuvenation intervals. Performance screen only; not convergence or formal800-step timing.',
                      'runtime_files':inventory(runtime)}
                write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'settings':settings,'cases':cases,'repeats':1,
               'notes':'2models x2dtypes x2sequence lengths x4variants. Fresh BF16 seq256 controls. Big bucket2048MiB keeps350M FP32 in one bucket; previous1024MiB would split it.'})
    print({'cases':len(cases),'gpu_execution':False})

if __name__=='__main__':main()
