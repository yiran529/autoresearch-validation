"""CPU acceptance checks for graph-only AutoResearch timing with passive recording."""
import ast
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch

from glval import bridge
from glval.common import VENDOR

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'campaigns/powersgd-graph-timing-20261118/prepare.py'


class GraphOnlyTiming(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'Graph-only recorded timing campaign is not implemented')
        self.scope = runpy.run_path(str(SCRIPT))

    def test_queue_cannot_include_dense_or_original_method(self):
        cases = self.scope['definitions']()
        self.assertEqual([(c['precision'], c['repetition']) for c in cases],
                         [('bf16', 1), ('fp32', 1), ('fp32', 2), ('bf16', 2)])
        for case in cases:
            self.assertEqual(case['method_id'], 'powersgd-r32-graph-v1')
            self.assertEqual(case['seed'], 1243)
            args = case['recipe']['args']
            self.assertTrue(args['record_bucket_layout'])
            self.assertFalse(args['strict_blocking_communication'])
            self.assertEqual((args['warmup_iterations'], args['measured_iterations']), (101, 400))
            self.assertEqual((args['batch_size'], args['optimizer'], args['ddp_bucket_cap_mb']), (1, 'adamw', 1024))
            self.assertEqual(case['recipe']['checkpoint_policy'], 'none')
            self.assertEqual(case['recipe']['tracking']['mode'], 'disabled')

    def test_observer_preserves_inner_future_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'timing.py'
            path.write_text((VENDOR / 'c4/table_v_timing.py').read_text())
            self.scope['patch_recording'](path)
            tree = ast.parse(path.read_text())
            names = {'BucketRecordingHookState', 'bucket_recording_hook', 'register_timing_comm_hook'}
            selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
            future_import = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
            namespace = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future_import, *selected], type_ignores=[])), str(path), 'exec'), namespace)

        class Future:
            def wait(self): raise AssertionError('Passive recording must never wait')
        class Tensor:
            def numel(self): return 64
            def element_size(self): return 2
        class Bucket:
            def buffer(self): return Tensor()
        class Model:
            def register_comm_hook(self, state, hook): self.state, self.hook = state, hook
        result = Future()
        inner_state = object()
        calls = []
        def inner(state, bucket):
            self.assertIs(state, inner_state)
            calls.append(bucket)
            return result
        def attach(model, process_group, args, optimizer=None):
            model.register_comm_hook(inner_state, inner)
        model = Model()
        args = types.SimpleNamespace(record_bucket_layout=True, strict_blocking_communication=False)
        with patch.object(bridge, 'register_comm_hook_for_ddp_model', side_effect=attach):
            observer = namespace['register_timing_comm_hook'](model, object(), args, object())
        observer.begin_step(True)
        bucket = Bucket()
        self.assertIs(model.hook(model.state, bucket), result)
        self.assertEqual(calls, [bucket])
        self.assertEqual(observer.end_step(), {'bucket_count': 1, 'bucket_bytes': [128]})
        self.assertEqual(model.register_comm_hook.__func__, Model.register_comm_hook)


if __name__ == '__main__': unittest.main()
