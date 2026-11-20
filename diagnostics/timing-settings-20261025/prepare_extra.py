"""Add the missing350M four-card default-bucket Dense reference, without replacing cases."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.build import build_runtime

HERE=Path(__file__).resolve().parent

def main():
    path=HERE/'cases.json'
    manifest=read_json(path)
    assert len(manifest['cases'])==40
    assert not (HERE/'cases-original40.json').exists()
    old_hash=file_hash(path)
    write_json(HERE/'cases-original40.json',manifest)
    reference=next(c for c in manifest['cases'] if c['id']=='350m-adam-8-default-dense')
    case=deepcopy(reference)
    case['id']='350m-adam-4-default-dense'
    case['setting'].update(id='350m-adam-4-default',world=4)
    case['world_size']=4
    directory=HERE/'cases'/case['id'];runtime=directory/'runtime'
    recipe=read_json(Path(reference['runtime'])/'recipe.json');recipe['world_size']=4
    build_runtime(runtime,ROOT/'methods/dense/method.json',recipe)
    case.update(runtime=str(runtime),output=str(directory/'output'),runtime_files=inventory(runtime))
    write_json(directory/'case.json',case)
    manifest['settings'].append(case['setting']);manifest['cases'].append(case)
    write_json(path,manifest)
    write_json(HERE/'extra-reference.json',{'reason':'Compare large-bucket compression against default-bucket Dense also on4GPUs, rather than only a disadvantaged big-bucket Dense.',
              'original_manifest_hash':old_hash,'extended_manifest_hash':file_hash(path),'case':case['id'],
              'execution':'Original controller has loaded40 cases; execute this extra case explicitly after it completes.'})
    print({'new_case':case['id'],'gpu_execution':False})

if __name__=='__main__':main()
