"""Independent setting sweep; frozen method semantics, no original-project edits."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT, read_json, write_json, inventory
from glval.build import build_runtime

HERE = Path(__file__).resolve().parent

def main():
    # Four-card runs use GPUs 4-7; eight-card runs use all GPUs, sequentially.
    settings = [
        dict(id='130m-muon-4-default', model='130m', optimizer='muon', world=4, bucket=None, checkpoint=True, batch=1),
        dict(id='130m-muon-8-default', model='130m', optimizer='muon', world=8, bucket=None, checkpoint=True, batch=1),
        dict(id='130m-adam-4-default', model='130m', optimizer='adamw', world=4, bucket=None, checkpoint=False, batch=1),
        dict(id='130m-adam-8-default', model='130m', optimizer='adamw', world=8, bucket=None, checkpoint=False, batch=1),
        dict(id='130m-adam-8-big', model='130m', optimizer='adamw', world=8, bucket=1024, checkpoint=False, batch=1),
        dict(id='350m-adam-4-big', model='350m', optimizer='adamw', world=4, bucket=1024, checkpoint=False, batch=1),
        dict(id='350m-adam-8-big', model='350m', optimizer='adamw', world=8, bucket=1024, checkpoint=False, batch=1),
        dict(id='350m-adam-8-default', model='350m', optimizer='adamw', world=8, bucket=None, checkpoint=False, batch=1),
    ]
    methods = [('dense','dense'), ('n23','n023-engineer-maxu'), ('n31','n031-engineer-maxu'),
               ('n20','n020-engineer-maxu'), ('greedy','greedylore-engineer-streamfix')]
    cases=[]
    for index, setting in enumerate(settings):
        recipe=read_json(ROOT/'recipes/llama130m-muon-timing-bsz1-local-v1.json')
        recipe.update(world_size=setting['world'], compression_start_step=2,
                      tracking={'mode':'disabled','name':'setting-diagnostic'})
        args=recipe['args']
        args.update(model_config='c4/configs/llama_'+setting['model']+'.json',
                    optimizer=setting['optimizer'], activation_checkpointing=setting['checkpoint'],
                    batch_size=setting['batch'], ddp_bucket_cap_mb=setting['bucket'],
                    warmup_iterations=21, measured_iterations=200)
        if setting['optimizer']=='adamw':
            args.update(lr=.0025,beta1=.9,beta2=.999,eps=1e-8)
        # Rotate methods between settings to avoid always measuring Dense on a colder device.
        rotated=methods[index%len(methods):]+methods[:index%len(methods)]
        for suffix, method in rotated:
            name=setting['id']+'-'+suffix
            directory=HERE/'cases'/name
            runtime=directory/'runtime'
            build_runtime(runtime,ROOT/'methods'/method/'method.json',deepcopy(recipe))
            case={'id':name,'setting':setting,'model':setting['model'],'method':method,
                  'runtime':str(runtime),'output':str(directory/'output'),
                  'bucket_mb':setting['bucket'],'local_muon':False,'no_ddp':False,
                  'world_size':setting['world'],'batch_size':setting['batch'],'sequence_length':256,
                  'profile':False,'profile_steps':0,'warmup':21,'measured':200,
                  'scope':'Cached synthetic tokens; compression starts2; updates22-221 measure one complete200-step refresh cycle. Performance only, no quality claim.',
                  'runtime_files':inventory(runtime)}
            write_json(directory/'case.json',case);cases.append(case)
    write_json(HERE/'cases.json',{'settings':settings,'cases':cases,'repeats':1})
    print({'cases':len(cases),'gpu_execution':False})

if __name__=='__main__':main()
