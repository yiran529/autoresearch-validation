"""N031 pooled-quota worker-local Top-k with explicit offset/value messages.

Prior art: Sparsified SGD with Memory (Stich et al., NeurIPS 2018).
This tests block-constrained sparsification with the pinned AdamW/Muon protocol;
it does not import SGD convergence guarantees for those optimizers.
"""
import torch
import torch.distributed as dist
from comm_hooks import default_hooks


def encode_blocks(h, k, dtype):
    """Return int16 offsets, native values, and actual local FP32 decode.

    h is a padded (groups, B) FP32 array, B <= 32768. Ties use torch.topk's local
    choice: workers need not agree on supports. Offsets accompany every value.
    """
    indices = h.abs().topk(k, dim=1, sorted=False).indices
    values = h.gather(1, indices).to(dtype)
    decoded = torch.zeros_like(h).scatter_(1, indices, values.float())
    return indices.to(torch.int16), values, decoded


def block_sparse_hook(state, bucket):
    state.maybe_accumulate_momentum_on_bucket(bucket)
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world = dist.get_world_size(group)
    buffer = bucket.buffer()
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group, buffer)
    block_size = state.sparse_block_size
    if block_size not in (512, 1024, 2048, 4096, 8192, 16384, 32768) or state.sparse_k != block_size // 32:
        raise ValueError('N031 requires power-of-two pooled groups and fixed 1/32 density')
    from comm_hooks.subspace_hook import _should_compress_subspace
    error = state.error_dict.get(bucket.index())
    if error is None:
        error = torch.zeros_like(buffer, dtype=torch.float32)
        state.error_dict[bucket.index()] = error
    if error.numel() != buffer.numel():
        raise ValueError('Sparse EF bucket layout changed after compression began')
    sparse, dense, blocks = [], [], []
    offset = 0
    for tensor, param in zip(bucket.gradients(), bucket.parameters()):
        n = tensor.shape[0] if tensor.ndim else 1
        m = tensor.numel() // n
        eligible = _should_compress_subspace(
            n, m, min(n, m, state.matrix_approximation_rank), state.min_compression_rate
        )[0]
        name = state.param_to_name[param]
        eligible = eligible and not any(tag in name for tag in state.uncompressed_names)
        size = tensor.numel()
        state.total_bit_before_compression += size * tensor.element_size() * 8
        if eligible:
            # Old EF is dead after augmentation. Keep FP32-first casting.
            h = error[offset:offset + size]
            torch.add(tensor.reshape(-1).float(), h, out=h)
            pad = (-size) % block_size
            if pad:
                h = torch.nn.functional.pad(h, (0, pad))
            blocks.append(h.view(-1, block_size))
            sparse.append((tensor, offset, size, h.numel() // block_size))
        else:
            dense.append(tensor)
            error[offset:offset + size].zero_()
        offset += size
    # Identical tensor eligibility/layout gives identical collective shapes,
    # regardless of the worker-local support choices or gradient values.
    if dense:
        packed_dense = torch.cat([t.reshape(-1) for t in dense])
        # Encoding does not consume the independent dense payload.
        # All ranks retain dense/offset/value collective order.
        dense_work = dist.all_reduce(packed_dense, group=group, async_op=True)
        state.total_bit_after_compression += packed_dense.numel() * packed_dense.element_size() * 8
    if blocks:
        h = torch.cat(blocks, dim=0)
        if buffer.is_cuda and buffer.dtype in (torch.bfloat16, torch.float32):
            from comm_hooks.block_sparse_cuda import residual_
            selected = h.abs().topk(state.sparse_k, dim=1, sorted=False).indices
            values = h.gather(1, selected).to(buffer.dtype)
            indices = selected.to(torch.int16)
            residual = residual_(h, indices, values)
        else:
            indices, values, local_decoded = encode_blocks(h, state.sparse_k, buffer.dtype)
            residual = h - local_decoded
        # Local EF uses the actual native-value reconstruction, before averaging.
        pos = 0
        for tensor, start, size, count in sparse:
            error[start:start + size].copy_(residual[pos:pos + count].reshape(-1)[:size])
            pos += count
        indices = indices.contiguous()
        values = values.contiguous()
        # Gloo/NCCL do not support int16 collectives: transport its exact
        # two-byte representation with the same byte AllGather as the parent.
        wire_indices = indices.view(torch.uint8)
        gathered_indices = [torch.empty_like(wire_indices) for _ in range(world)]
        gathered_values = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered_indices, wire_indices, group=group)
        dist.all_gather(gathered_values, values, group=group)
        if buffer.is_cuda and buffer.dtype in (torch.bfloat16, torch.float32):
            from comm_hooks.block_sparse_cuda import aggregate
            mean = aggregate([i.view(torch.int16) for i in gathered_indices], gathered_values, block_size)
        else:
            mean = torch.zeros_like(h)
            for worker_indices, worker_values in zip(gathered_indices, gathered_values):
                # Each block has unique offsets; no atomic scatter-add is needed.
                local_decoded.zero_().scatter_(1, worker_indices.view(torch.int16).long(), worker_values.float())
                mean.add_(local_decoded)
            mean.div_(world)
        pos = 0
        for tensor, start, size, count in sparse:
            tensor.copy_(mean[pos:pos + count].reshape(-1)[:size].view_as(tensor))
            pos += count
        # Logical outgoing payload includes indices. AllGather wire traffic and
        # receive/decode amplification by world size are separate from this count.
        state.total_bit_after_compression += indices.numel() * indices.element_size() * 8 + values.numel() * values.element_size() * 8
    if dense:
        dense_work.wait()
        packed_dense.div_(world)
        pos = 0
        for tensor in dense:
            tensor.copy_(packed_dense[pos:pos + tensor.numel()].view_as(tensor))
            pos += tensor.numel()
    if buffer.is_cuda:
        torch.cuda.synchronize(buffer.device)
    state.maybe_increase_iter(bucket)
    result = torch.futures.Future()
    result.set_result(buffer)
    return result
