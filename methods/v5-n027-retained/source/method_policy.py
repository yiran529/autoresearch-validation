"""Frozen guarded coordinate/H16 pooled SparseQ4-EF policy; only communication fields are changed."""


def configure_compression(args, workload):
    if args.compressor == 'none':
        return args
    args.compressor = 'sparse_q4_ef'
    args.use_error_feedback = 'ef14'
    if workload == 'resnet18_cifar100_muon':
        args.start_compress_iter = 4000
    # Preserve the supplied language/timing dense prefix (1000/measurement warmup).
    args.update_proj_gap = 1
    return args
