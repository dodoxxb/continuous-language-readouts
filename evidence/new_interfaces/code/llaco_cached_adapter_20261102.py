"""Optional inference-only cache for the released LLaCO directional adapter.

Uses the algebraically equivalent dense FP32 implementation tested in
new_interfaces_llaco_20261102.py. It trades an additional 4 GiB at V=131072,
D=4096 for fewer conversions/kernel launches. This is a component optimization;
full-model quality, KV-cache capacity and end-to-end latency are not validated.
Do not install while training, loading checkpoints, or moving model devices.
"""
import types
import torch
from new_interfaces_llaco_20261102 import DenseAdapter

def install_cached_adapter(adapter, *, normalize, reserve_gib=2.0):
    """Patch only forward; return a restoration callback. State dict unchanged."""
    if adapter.training or adapter.compute_dtype!=torch.float32:
        raise ValueError('Requires eval mode and released FP32 adapter computation')
    if hasattr(adapter,'_elf_dense_cache'):
        raise ValueError('Adapter cache already installed')
    emb=adapter.geometry.E_geometry
    required=emb.numel()*8
    if emb.is_cuda:
        free,_=torch.cuda.mem_get_info(emb.device)
        if free<required+reserve_gib*2**30:
            raise RuntimeError('Insufficient headroom for exact adapter cache')
    with torch.inference_mode():cache=DenseAdapter(adapter,normalize)
    versions=(adapter.log_beta._version,adapter.bias._version,emb._version)
    original_forward=adapter.forward
    original_keys=set(adapter.state_dict())
    # Nonpersistent buffers follow the module lifecycle but are not checkpoints.
    adapter.register_buffer('_elf_values',cache.values,persistent=False)
    adapter.register_buffer('_elf_directions',cache.directions,persistent=False)
    adapter._elf_dense_cache=cache
    def forward(self,state,*,return_stats=False):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Cached adapter is inference-only; use torch.inference_mode()')
        if state.device!=self._elf_values.device:
            raise RuntimeError('Reinstall cache after moving the model')
        if (self.log_beta._version,self.bias._version,self.geometry.E_geometry._version)!=versions or self._elf_values.data_ptr()!=cache.values.data_ptr():
            raise RuntimeError('Reinstall cache after changing weights or moving model buffers')
        p=cache.probabilities(state)
        output=(p@cache.values).reshape_as(state).to(state.dtype)
        if not return_stats:return output
        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
        return output,{'adapter_entropy':entropy.reshape(state.shape[:-1]),'beta':self.beta.detach()}
    adapter.forward=types.MethodType(forward,adapter)
    assert set(adapter.state_dict())==original_keys
    def restore():
        adapter.forward=original_forward
        del adapter._elf_values,adapter._elf_directions,adapter._elf_dense_cache
        assert set(adapter.state_dict())==original_keys
    return restore
