"""CPU checks for the authorized optimized 350M restart and complete hook saves."""
import ast
import copy
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

from glval.common import read_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'campaigns/powersgd-graph-350m-rolling10k-20261118'


class GraphRollingContract(unittest.TestCase):
    def test_full_protocol_and_smoke_only_allow_authorized_changes(self):
        script = CAMPAIGN / 'prepare.py'
        self.assertTrue(script.exists(), 'Rolling10k graph campaign is not implemented')
        scope = runpy.run_path(str(script))
        for phase in ('smoke', 'full'):
            recipe, execution = scope['definition'](phase)
            scope['validate_contract'](recipe, execution, phase)
            self.assertEqual(execution['tracking']['mode'], 'disabled')
            self.assertEqual(recipe['checkpoint_policy']['every_updates'], 4 if phase == 'smoke' else 10000)
            self.assertEqual(recipe['args']['save_every'], recipe['checkpoint_policy']['every_updates'])
            for key, value in [('batch_size', 1), ('optimizer', 'adamw'), ('lr', .01), ('num_training_steps', 999)]:
                changed = copy.deepcopy(recipe)
                changed['args'][key] = value
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    scope['validate_contract'](changed, execution, phase)
            for key, value in [('every_updates', 20000), ('root', '/tmp/elsewhere')]:
                changed = copy.deepcopy(recipe)
                changed['checkpoint_policy'][key] = value
                with self.assertRaises(ValueError):
                    scope['validate_contract'](changed, execution, phase)
        recipe, execution = scope['definition']('full')
        baseline = read_json(ROOT / 'recipes/llama350m-muon-full-offline-v5.json')
        self.assertEqual(recipe['args'], {**baseline['args'], 'save_every': 10000})
        execution['tracking']['mode'] = 'online'
        with self.assertRaises(ValueError): scope['validate_contract'](recipe, execution, 'full')

    def test_power_sgd_slots_and_dynamic_fields_are_both_saved(self):
        path = CAMPAIGN / 'checkpoint_state.py'
        self.assertTrue(path.exists(), 'PowerSGD slot checkpoint extraction is not implemented')
        snapshot = runpy.run_path(str(path))['snapshot_hook']
        tree = ast.parse((ROOT / 'methods/powersgd-r32-graph-v1/source/comm_hooks/powerSGD_hook.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PowerSGDState')
        slots = ast.literal_eval(next(n.value for n in cls.body if isinstance(n, ast.Assign)
                                    and any(isinstance(t, ast.Name) and t.id == '__slots__' for t in n.targets)))
        class Base:
            pass
        Slotted = type('Slotted', (Base,), {'__slots__': slots})
        Scoped = type('Scoped', (Slotted,), {})
        state = Scoped()
        for key in slots: setattr(state, key, 'slot-' + key)
        state.error_dict = {1: 'EF'}
        state.p_memory_dict = {1: 'P'}
        state.q_memory_dict = {1: 'Q'}
        state.compressor_name = 'PowerSGD'
        state.excluded_parameter_ids = {123}
        state.param_state = object()
        state.param_to_name = object()
        self.assertNotIn('error_dict', vars(state), 'Fixture must reproduce slots omission')
        result = snapshot(state, lambda value: value)
        for key in slots:
            if key != 'process_group': self.assertEqual(result[key], getattr(state, key))
        self.assertEqual(result['compressor_name'], 'PowerSGD')
        for key in ('process_group', 'param_state', 'param_to_name', 'excluded_parameter_ids'):
            self.assertNotIn(key, result)
        self.assertEqual(state.error_dict, {1: 'EF'})

    def test_dense_only_buckets_may_have_empty_pq_but_compressed_buckets_may_not(self):
        scope = runpy.run_path(str(CAMPAIGN / 'verify_checkpoint.py'))
        self.assertIn('validate_hook_memory', scope,
                      'Checkpoint validation must distinguish dense-only buckets')
        class Tensor:
            device = SimpleNamespace(type='cpu')
            def __init__(self, size): self.size = size
            def numel(self): return self.size
        hook = {key: {i: Tensor(100) for i in range(23)}
                for key in ('error_dict', 'p_memory_dict', 'q_memory_dict')}
        for bucket, size in ((0, 32768000), (22, 32769024)):
            hook['error_dict'][bucket] = Tensor(size)
            hook['p_memory_dict'][bucket] = Tensor(0)
            hook['q_memory_dict'][bucket] = Tensor(0)
        check = scope['validate_hook_memory']
        counts = check(hook, lambda v: isinstance(v, Tensor))
        self.assertEqual(counts['p_memory_dict'], 2100)
        for key, bucket, value in [('p_memory_dict', 3, Tensor(0)),
                                   ('p_memory_dict', 0, Tensor(100)),
                                   ('error_dict', 22, Tensor(1)),
                                   ('q_memory_dict', 2, object())]:
            bad = copy.deepcopy(hook)
            bad[key][bucket] = value
            with self.subTest(key=key, bucket=bucket), self.assertRaises(AssertionError):
                check(bad, lambda v: isinstance(v, Tensor))
        bad = copy.deepcopy(hook)
        del bad['q_memory_dict'][7]
        with self.assertRaises(AssertionError): check(bad, lambda v: isinstance(v, Tensor))


if __name__ == '__main__': unittest.main()
