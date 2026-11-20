from campaign_fixtures import build_fixture
import ast
import json
from pathlib import Path
import unittest
from glval.common import inventory
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'diagnostics/head-emb-timing-20261026'

class HeadEmbTiming(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global HERE
        cls.original_directory = HERE
        cls.fixture, HERE = build_fixture(HERE / 'prepare.py', reuse_variant=True)

    @classmethod
    def tearDownClass(cls):
        global HERE
        HERE = cls.original_directory
        cls.fixture.cleanup()


    def test_variant_changes_coverage_only(self):
        parent = ROOT / 'methods/greedylore-engineer-streamfix'
        variant = ROOT / 'methods/greedylore-engineer-streamfix-head-emb-v1'
        a = json.loads((parent / 'method.json').read_text())
        b = json.loads((variant / 'method.json').read_text())
        self.assertEqual(a['source_commit'], b['source_commit'])
        self.assertEqual(a['parameters'], b['parameters'])
        self.assertEqual(inventory(parent / 'source'), inventory(variant / 'source'))
        tree = ast.parse((variant / 'adapter.py').read_text())
        lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
        self.assertEqual(len(lambdas), 1)
        self.assertEqual(ast.literal_eval(lambdas[0].body), [])
        self.assertIn('finally:', (variant / 'adapter.py').read_text())

    def test_matched_refresh_inclusive_design(self):
        cases = json.loads((HERE / 'cases.json').read_text())['cases']
        self.assertEqual(len(cases), 9)
        settings = {}
        for c in cases:
            settings.setdefault(c['setting']['id'], []).append(c)
            self.assertEqual(c['world_size'], 4)
            self.assertEqual(c['batch_size'], 1)
            self.assertEqual(c['warmup'], 21)
            self.assertEqual(c['measured'], 400)
            self.assertEqual(c['bucket_mb'], 2048)
            recipe = json.loads((Path(c['runtime']) / 'recipe.json').read_text())
            self.assertEqual(recipe['tracking']['mode'], 'disabled')
            self.assertEqual(recipe['args']['optimizer'], 'adamw')
            self.assertEqual(recipe['args']['lr'], 0.0025)
            self.assertFalse(recipe['args']['activation_checkpointing'])
            self.assertEqual(recipe['compression_start_step'], 2)
            self.assertEqual(inventory(Path(c['runtime'])), c['runtime_files'])
        for group in settings.values():
            self.assertEqual({c['variant'] for c in group}, {'dense', 'greedy_exclude', 'greedy_include'})
            self.assertEqual(len({json.dumps(c['setting'], sort_keys=True) for c in group}), 1)
