"""Create isolated diagnostic source variants. Original methods remain immutable."""
import ast
import shutil
import subprocess
from pathlib import Path
from glval.common import ROOT,write_json,file_hash

HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'methods/greedylore-engineer/source'

def main():
    variants=HERE/'variants'
    variants.mkdir(exist_ok=False)
    original=(SOURCE/'comm_hooks/subspace_hook.py').read_text()
    marker='    def unpack_uncompressed_tensors_and_allreduce_rs(fut):\n'
    fixed=original.replace(marker,
        '    # The dense collective was launched before R/EF production. Its Future\n'
        '    # cannot order those later writes onto the NCCL callback stream.\n'
        '    producer_ready = torch.cuda.Event()\n'
        '    producer_ready.record(torch.cuda.current_stream(device))\n\n'+marker+
        '        torch.cuda.current_stream(device).wait_event(producer_ready)\n',1)
    # Preserve the real SVD exception instead of raising UnboundLocalError in its logger.
    fixed='\n'.join(line for line in fixed.split('\n') if not any(
        fragment in line for fragment in ['logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_U:',
                                        'logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_Vh:']))
    start=original.index('    # Scores have completed their collective.')
    end=original.index('    # Create Rs that point',start)
    launch=original[start:end]
    serial=original[:start]+original[end:]
    serial=serial.replace(marker,launch+marker,1)
    research=subprocess.check_output(['git','-C','/home/wyr/coral-experiments/root-reuse-20261022/run/repo',
                 'show','d6009710690cd4f1985ec6f00f03a94e5acf504a:comm_hooks/subspace_hook.py'],text=True)
    for name,text in [('engineer',original),('eventfix',fixed),('serial',serial),('research',research)]:
        ast.parse(text)
        (variants/(name+'.py')).write_text(text)
    repo=HERE/'repo'
    shutil.copytree(SOURCE,repo)
    (repo/'comm_hooks/subspace_hook.py').write_text(fixed)
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Validation Agent','-c','user.email=validation@localhost',
                    'commit','-qm','Fix GreedyLoRE EF14 producer-to-callback dependency and preserve SVD errors'],check=True)
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    write_json(HERE/'source-provenance.json',{'parent_engineer':'382bfae897a4db84af45e021ed1952bab7584445',
        'research_reference':'d6009710690cd4f1985ec6f00f03a94e5acf504a','fix_commit':sha,
        'variants':{p.name:file_hash(p) for p in variants.glob('*.py')},
        'parameter_changes':False,'training_loop_changes':False,
        'fix':'Record CUDA producer event after R and EF production; callback stream waits before reading/unpacking. Remove logger reads of undefined SVD outputs.'})
    print({'fix_commit':sha,'variants':str(variants),'gpu_execution':False})

if __name__=='__main__':main()
