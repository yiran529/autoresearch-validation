from campaign_fixtures import build_fixture
import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'campaigns/resnet50-adamw-five-methods-full-20261026'

class ResNet50AdamWCampaign(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global CAMPAIGN
        cls.original_directory = CAMPAIGN
        cls.fixture, CAMPAIGN = build_fixture(CAMPAIGN / 'prepare.py', reuse_variant=False)

    @classmethod
    def tearDownClass(cls):
        global CAMPAIGN
        CAMPAIGN = cls.original_directory
        cls.fixture.cleanup()

    def test_all_methods_share_adamw_and_budget(self):
        cases=json.loads((CAMPAIGN/'cases.json').read_text())['cases']
        self.assertEqual(len(cases),10)
        for case in cases:
            with self.subTest(case=case['id']):
                args=case['recipe']['args']
                self.assertEqual(args['optimizer'],'adamw')
                self.assertEqual(args['lr'],0.001)
                self.assertEqual(args['weight-decay'],0.0005)
                self.assertEqual(args['batch-size'],32)
                self.assertFalse(any(k.startswith('muon_') for k in args))
                if case['phase']=='full':
                    self.assertEqual(args['epochs'],40)
                    self.assertEqual(case['recipe']['completion']['update_step'],15640)
                tree=ast.parse((Path(case['runtime'])/'pytorch-cifar/train_ddp.py').read_text())
                optimizers=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='optimizer' for t in n.targets)]
                self.assertEqual(len(optimizers),1)
                call=optimizers[0].value
                self.assertEqual(ast.unparse(call.func),'torch.optim.AdamW')
                kwargs={kw.arg:ast.unparse(kw.value) for kw in call.keywords}
                self.assertEqual(kwargs['betas'],'(0.9, 0.999)')
                self.assertEqual(float(kwargs['eps']),1e-8)
                src=ast.unparse(tree)
                self.assertNotIn('communication_bits_stats()',src)
                self.assertNotIn('build_muon_optimizer',src)
                self.assertIn('ResNet(Bottleneck, [3, 4, 6, 3]',src)
                self.assertIn("choices=['adamw']",src)
                self.assertEqual(case['expected_parameter_count'],23520842)
