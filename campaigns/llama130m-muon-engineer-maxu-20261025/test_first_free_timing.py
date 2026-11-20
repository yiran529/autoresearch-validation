import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('first_free',Path(__file__).with_name('first_free_timing.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Checks(unittest.TestCase):
    def test_first_group_can_be_either_group(self):
        self.assertEqual(module.wait_first([None,12]),0)
        self.assertEqual(module.wait_first([11,None]),1)
        with patch.object(module.select,'select',return_value=([12],[],[])):
            self.assertEqual(module.wait_first([11,12]),1)
        with patch.object(module.select,'select',return_value=([11],[],[])):
            self.assertEqual(module.wait_first([11,12]),0)

    def test_timing_fixed_allocation_and_dense_last(self):
        hardware=[{'uuid':'g4','index':'4'},{'uuid':'g5','index':'5'},
                  {'uuid':'g6','index':'6'},{'uuid':'g7','index':'7'}]
        plan={'jobs':[{'method_id':m} for _ in range(3) for m in ['greedy','n23','n31','n20','dense']]}
        with patch.object(module,'verify_plan',return_value=plan),patch.object(module,'write_json'), \
             patch.object(module,'run_plan') as run:
            module.run_timing({'timing_run':'fake'},hardware,{})
        self.assertEqual(run.call_count,2)
        self.assertEqual(run.call_args_list[0].kwargs['only_methods'],['greedy','n23','n31','n20'])
        self.assertEqual(run.call_args_list[1].kwargs['only_methods'],['dense'])
        for call in run.call_args_list:
            self.assertEqual(call.args[1],'g4,g5,g6,g7')
            self.assertIs(call.kwargs['hardware_allocation'],hardware)

    def test_inherited_reservation_transfers_without_stalling_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'locks').mkdir()
            hardware=[{'uuid':f'g{i}'} for i in range(4)]
            code='import pathlib,fcntl,sys; p=pathlib.Path(sys.argv[1]); files=[(p/("g"+str(i)+".lock")).open("a") for i in range(4)]; [fcntl.flock(f,fcntl.LOCK_EX) for f in files]; print("ready",flush=True); sys.stdin.read(1)'
            with subprocess.Popen([sys.executable,'-c',code,str(root/'locks')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as owner:
                self.assertEqual(owner.stdout.readline().strip(),'ready')
                previous={'pid':owner.pid,'ticks':module.proc(owner.pid)['ticks']}
                with patch.object(module,'ROOT',root):
                    with module.campaign_lease(hardware,previous) as lease:
                        self.assertTrue(lease['inherited_lock_bridge'])
                        owner.stdin.write('x');owner.stdin.flush()
                        owner.wait(timeout=5)

    def test_busy_gpu_blocks_timing(self):
        hardware=[{'uuid':str(i)} for i in range(4)]
        current=[{'uuid':str(i),'memory_mib':'0','utilization':'0'} for i in range(4)]
        with patch.object(module,'query_gpus',return_value=current), \
             patch.object(module.subprocess,'check_output',return_value='0, 123\n'):
            with self.assertRaises(RuntimeError):module.check_idle(hardware)

    def test_orphan_evidence_does_not_invent_exit_code(self):
        state={'jobs':{'n20':{'status':'running','pid':123,'started_at':1}}}
        plan={'jobs':[{'id':'n20','method_id':'n20'}],'recipe':{}}
        with patch.object(module,'verify_plan',return_value=plan), \
             patch.object(module,'read_json',return_value=state), \
             patch.object(module,'inspect_job',return_value={'status':'verified','metrics':{'loss':3.1}}), \
             patch.object(module,'write_json'):
            module.finalize_orphan('fake','n20')
        self.assertEqual(state['jobs']['n20']['status'],'finished_exit_unknown')
        self.assertIsNone(state['jobs']['n20']['returncode'])
        self.assertFalse(state['jobs']['n20']['process_exit_verified'])

if __name__=='__main__':unittest.main(verbosity=2)
