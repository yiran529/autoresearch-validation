from campaign_fixtures import build_fixture
"""Static checks protect actual model construction, shared recipe and source identity."""
import ast
from pathlib import Path
import unittest
from glval.common import ROOT,read_json,inventory

HERE=ROOT/'campaigns/resnet50-five-methods-full-20261026'

class ResNet50CampaignTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global HERE
        cls.original_directory = HERE
        cls.fixture, HERE = build_fixture(HERE / 'prepare.py', reuse_variant=False)

    @classmethod
    def tearDownClass(cls):
        global HERE
        HERE = cls.original_directory
        cls.fixture.cleanup()

    def test_actual_builder_uses_bottleneck50_for_all_cases(self):
        for case in read_json(HERE/'cases.json')['cases']:
            tree=ast.parse((Path(case['runtime'])/case['driver']).read_text())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_model')
            block=object();scope={'Bottleneck':block,'ResNet':lambda b,layers,num_classes:(b,layers,num_classes)}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<builder-test>','exec'),scope)
            self.assertEqual(scope['build_model']('cifar10'),(block,[3,4,6,3],10))
            self.assertEqual(scope['build_model']('cifar100'),(block,[3,4,6,3],100))

    def test_full_budget_optimizer_and_data_match_resnet18(self):
        original=read_json(ROOT/'recipes/resnet18-cifar10-muon-full-v1.json')
        full=[c for c in read_json(HERE/'cases.json')['cases'] if c['phase']=='full']
        self.assertEqual(len(full),5)
        for case in full:
            self.assertEqual(case['recipe']['args'],original['args'])
            self.assertEqual(case['recipe']['completion'],original['completion'])
            self.assertEqual(case['recipe']['compression_start_step'],500)
            self.assertEqual(case['recipe']['world_size'],4)

    def test_frozen_method_code_matches_existing_resnet_variant(self):
        for case in read_json(HERE/'cases.json')['cases']:
            runtime=Path(case['runtime']);method=read_json(runtime/'method.json')
            source=ROOT/'methods'/case['method_id']
            self.assertEqual(method,read_json(source/'method.json'))
            self.assertEqual(inventory(runtime/'comm_hooks'),inventory((ROOT/'vendor/greedy_lore' if method.get('builtin') else source/'source')/'comm_hooks'))

if __name__=='__main__':unittest.main()
