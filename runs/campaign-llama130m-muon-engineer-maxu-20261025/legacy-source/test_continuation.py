import importlib.util
from pathlib import Path
import subprocess
import select
import os
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('continuation',Path(__file__).with_name('continue.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Checks(unittest.TestCase):
    def test_pidfd_waits_for_exact_process(self):
        with subprocess.Popen(['python3','-c','pass']) as child:
            fd = os.pidfd_open(child.pid)
            try:
                self.assertTrue(select.select([fd],[],[],5)[0])
            finally:
                os.close(fd)

    def test_order_failure_preservation_and_dense_last(self):
        plans = {
            'q0':{'jobs':[{'id':x,'method_id':x} for x in ['greedy','n31','dense']]},
            'q1':{'jobs':[{'id':x,'method_id':x} for x in ['n23','n20']]},
            'timing':{'jobs':[{'id':x,'method_id':x} for x in ['greedy','n23','n31','n20','dense']]}}
        states = {r:{'jobs':{j['id']:{'status':'planned'} for j in p['jobs']}} for r,p in plans.items()}
        states['q0']['jobs']['greedy']['status']='failed'
        states['q1']['jobs']['n23']['status']='completed'
        calls=[]
        def fake_read(path):
            return states[Path(path).parent.name]
        def fake_run(run,selection,execute,only_methods,hardware_allocation):
            calls.append((run,list(only_methods)))
            for job in plans[run]['jobs']:
                if job['method_id'] in only_methods:
                    self.assertEqual(states[run]['jobs'][job['id']]['status'],'planned')
                    states[run]['jobs'][job['id']]['status']='completed'
        class Process:
            def __init__(self,target,args): self.target,self.args,self.exitcode=target,args,None
            def start(self): self.target(*self.args);self.exitcode=0
            def join(self): pass
        config={'training_queues':[{'run':'q0','gpus':'0,1,2,3'},{'run':'q1','gpus':'4,5,6,7'}],'timing_run':'timing'}
        with patch.object(module,'verify_plan',side_effect=lambda r:plans[r]), \
             patch.object(module,'read_json',side_effect=fake_read), \
             patch.object(module,'write_json'), \
             patch.object(module,'run_plan',side_effect=fake_run), \
             patch.object(module,'report',return_value={}), \
             patch.object(module,'gpu_lease',side_effect=lambda *_:nullcontext([{'uuid':'test'}])), \
             patch.object(module.multiprocessing,'get_context',return_value=SimpleNamespace(Process=Process)):
            module.execute_campaign(config,Path('/tmp/fake-campaign'))
        self.assertEqual(calls,[('q0',['n31']),('q1',['n20']),('q0',['dense']),
                                ('timing',['greedy','n23','n31','n20']),('timing',['dense'])])
        self.assertEqual(states['q0']['jobs']['greedy']['status'],'failed')

    def test_running_dependency_blocks_recovery(self):
        plan={'jobs':[{'id':'n','method_id':'n'}]}
        with patch.object(module,'verify_plan',return_value=plan), \
             patch.object(module,'read_json',return_value={'jobs':{'n':{'status':'running'}}}):
            with self.assertRaises(RuntimeError): module.remaining_methods('fake',False)

if __name__ == '__main__': unittest.main(verbosity=2)
