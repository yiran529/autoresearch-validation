import ast,json,tempfile,unittest
from pathlib import Path
from glval.checkpoints import publish
from glval.build import DriverBridge
from glval.common import VENDOR
from glval import checkpoints
class RollingCheckpointTest(unittest.TestCase):
 def test_shared_checkpoint_preserves_inherited_slots_and_dynamic_state(self):
  class Base:
   __slots__=('process_group','error_dict')
  class State(Base):pass
  state=State();state.process_group=object();state.error_dict={1:'EF'};state.q_memory_dict={1:'Q'}
  state.excluded_parameter_ids={123}
  snapshot=checkpoints.snapshot_hook(state,lambda value:value)
  self.assertEqual(snapshot,{'error_dict':{1:'EF'},'q_memory_dict':{1:'Q'}})
 def test_incomplete_replacement_keeps_old_checkpoint(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'old';old.mkdir();(old/'manifest.json').write_text('{}');(old/'model.pt').write_bytes(b'good')
   publish(old,root,20000);bad=root/'partial';bad.mkdir();(bad/'rank0.pt').write_bytes(b'partial')
   with self.assertRaises(ValueError):publish(bad,root,40000)
   self.assertEqual((root/'latest/model.pt').read_bytes(),b'good')
   (bad/'manifest.json').write_text('{}');publish(bad,root,40000)
   self.assertFalse((root/'step-00020000').exists());self.assertTrue((root/'latest/rank0.pt').exists())
 def test_all_rank_callback_after_optimizer_and_step_before_eval(self):
  tree=ast.parse((VENDOR/'c4/run_llama_pretraining.py').read_text());b=DriverBridge('llama',True);tree=b.visit(tree)
  self.assertEqual(b.checkpoint_injections,1);s=ast.unparse(ast.fix_missing_locations(tree))
  self.assertLess(s.index('optimizer.step()'),s.index('save_if_due(locals())'))
  self.assertLess(s.index('update_step += 1'),s.index('save_if_due(locals())'))
  self.assertNotIn('update_step % args.save_every',s)
