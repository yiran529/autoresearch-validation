"""No torch imports: protect the new model-name bridge and frozen parent identity."""
import argparse
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from glval.common import ROOT,read_json,inventory

PARENTS=['greedylore-engineer-streamfix','n023-engineer-maxu','n031-engineer-maxu','n020-engineer-maxu']

def adapter(parent):
    path=ROOT/'methods'/(parent+'-resnet-v1')/'adapter.py'
    spec=importlib.util.spec_from_file_location('resnet_adapter_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.MethodAdapter()

class ResNetAdapterTests(unittest.TestCase):
    def exercise(self,parent,fail=False):
        package=types.ModuleType('comm_hooks');utils=types.ModuleType('comm_hooks.utils');package.utils=utils
        original=lambda model:['legacy']
        utils.get_uncompress_names=original
        utils.add_comm_hook_args=lambda parser:parser.add_argument('--scale',type=float,default=1.)
        seen={}
        def register(model,group,args,optimizer=None):
            seen.update(names=utils.get_uncompress_names(model),args=vars(args),optimizer=optimizer)
            if fail:raise RuntimeError('registration failed')
            return 'registered'
        utils.register_comm_hook_for_ddp_model=register
        config=read_json(ROOT/'methods'/parent/'method.json')['parameters']
        with patch.dict(sys.modules,{'comm_hooks':package,'comm_hooks.utils':utils}):
            call=lambda:adapter(parent).attach(object(),'optimizer','group',argparse.Namespace(seed=1243,start_compress_iter=500),{'workload':'resnet','world_size':4},config)
            if fail:
                with self.assertRaises(RuntimeError):call()
            else:self.assertEqual(call(),'registered')
        self.assertIs(utils.get_uncompress_names,original)
        self.assertEqual(seen['names'],['linear'])
        self.assertEqual(seen['args']['seed'],1243)
        self.assertEqual(seen['args']['start_compress_iter'],500)
        self.assertEqual(seen['args']['optimizer'],'muon')
        for key,value in config.items():self.assertEqual(seen['args'][key],value)

    def test_all_registration_bridges_preserve_parameters(self):
        for parent in PARENTS:
            with self.subTest(parent=parent):self.exercise(parent)

    def test_registration_failure_restores_shared_helper(self):
        for parent in PARENTS:
            with self.subTest(parent=parent):self.exercise(parent,True)

    def test_variants_cannot_be_used_as_llama_adapters(self):
        for parent in PARENTS:
            with self.assertRaises(ValueError):adapter(parent).validate({'workload':'llama','world_size':4},{'compress_rank':32,'update_proj_gap':200})

    def test_candidate_sources_and_parameters_equal_frozen_parents(self):
        for parent in PARENTS:
            original=ROOT/'methods'/parent;variant=ROOT/'methods'/(parent+'-resnet-v1')
            a,b=read_json(original/'method.json'),read_json(variant/'method.json')
            self.assertEqual(a['source_commit'],b['source_commit']);self.assertEqual(a['parameters'],b['parameters'])
            self.assertEqual(inventory(original/'source'),inventory(variant/'source'))

if __name__=='__main__':unittest.main()
