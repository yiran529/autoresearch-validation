"""ResNet-only registration bridge; frozen candidate algorithms remain unchanged."""
import argparse


class MethodAdapter:
    def validate(self, context, config):
        if context['workload']!='resnet' or context['world_size']<2:
            raise ValueError('This variant requires distributed ResNet')
        if config['compress_rank']<1 or config['update_proj_gap']<1:
            raise ValueError('Invalid frozen method parameters')

    def attach(self, model, optimizer, process_group, args, context, config):
        from comm_hooks import utils
        parser=argparse.ArgumentParser(add_help=False)
        utils.add_comm_hook_args(parser)
        merged=vars(parser.parse_args([]));merged.update(vars(args));merged.update(config)
        merged.setdefault('optimizer','muon')
        original=utils.get_uncompress_names
        # The canonical CIFAR driver excludes its linear classifier. All conv
        # matrix views, EF, ranks, projections and collectives stay in source.
        utils.get_uncompress_names=lambda ddp_model:['linear']
        try:
            return utils.register_comm_hook_for_ddp_model(
                model,process_group,argparse.Namespace(**merged),optimizer=optimizer)
        finally:
            utils.get_uncompress_names=original

    def describe(self):
        return {'workload':'resnet','uncompressed_names':['linear'],
                'matrix_mapping':'Candidate native tensor.view(out_channels,-1) / flat block sparse',
                'implementation':'Frozen parent Engineer source; ResNet registration bridge only',
                'communication_volume':'Generic estimate unavailable'}
