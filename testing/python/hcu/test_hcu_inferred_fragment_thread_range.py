"""Natural-index Parallel loops must retain their fragment's thread domain."""

import pytest
import torch
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


TARGET = {"kind": "hcu", "mcpu": "gfx938"}


@pytest.mark.parametrize("offset,width,replicas", [(0, 64, 1), (64, 64, 1), (256, 256, 1), (128, 64, 2), (64, 1, 64)])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_inferred_consumer_fragment_thread_range(offset, width, replicas):
    """The offset=64 case used to silently return X instead of X+1."""
    if replicas == 1:
        layout = T.Fragment((width,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)
    else:
        layout = T.Fragment(
            (width,),
            replicate=replicas,
            forward_thread_fn=lambda i, rep: i + rep * width,
            forward_index_fn=lambda i: 0,
        )

    @T.prim_func
    def main(X: T.Tensor((width,), "float32"), Y: T.Tensor((width,), "float32")):
        with T.Kernel(1, threads=offset + width * replicas):
            a = T.alloc_fragment((width,), "float32")
            T.annotate_layout({a: layout})
            tx = T.get_thread_binding()
            if tx >= offset:
                T.copy(X, a, loop_layout=layout)
                for i in T.Parallel(width):
                    a[i] = a[i] + 1
                T.copy(a, Y, loop_layout=layout)

    kernel = tl.compile(main, out_idx=[1], target=TARGET)
    x = torch.arange(width, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(x), x + 1, atol=0, rtol=0)


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_unbound_annotation_reused_in_disjoint_domains():
    """Binding a loop candidate must not mutate the shared annotation."""
    layout = T.Fragment((64,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)

    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32"), Z: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_fragment((64,), "float32")
            T.annotate_layout({a: layout})
            tx = T.get_thread_binding()
            if tx < 64:
                T.copy(X, a, loop_layout=layout)
                for i in T.Parallel(64):
                    a[i] = a[i] + 1
                T.copy(a, Y, loop_layout=layout)
            if tx >= 64:
                # These threads initialize their own registers; there is no
                # implicit transfer between producer and consumer registers.
                T.copy(X, a, loop_layout=layout)
                for i in T.Parallel(64):
                    a[i] = a[i] + 2
                T.copy(a, Z, loop_layout=layout)

    kernel = tl.compile(main, out_idx=[1, 2], target=TARGET)
    x = torch.arange(64, device="cuda", dtype=torch.float32)
    y, z = kernel(x)
    torch.testing.assert_close(y, x + 1, atol=0, rtol=0)
    torch.testing.assert_close(z, x + 2, atol=0, rtol=0)


def _bound_fragment_kernel(incompatible):
    layout = T.Fragment((64,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)

    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_fragment((64,), "float32")
            tx = T.get_thread_binding()
            if tx >= 64:
                # Inference assigns a a bound [64,128) fragment here.
                for i in T.Parallel(64, loop_layout=layout):
                    a[i] = X[i]
            if incompatible:
                if tx < 64:
                    for i in T.Parallel(64):
                        a[i] = a[i] + 1
            else:
                # A wider execution context must preserve ownership rather
                # than rebind a to [0,128).
                for i in T.Parallel(64):
                    a[i] = a[i] + 1
            if tx >= 64:
                T.copy(a, Y, loop_layout=layout)

    return main


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_bound_fragment_in_wider_domain():
    kernel = tl.compile(_bound_fragment_kernel(False), out_idx=[1], target=TARGET)
    x = torch.arange(64, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(x), x + 1, atol=0, rtol=0)


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_bound_fragment_in_disjoint_domain_rejected():
    with pytest.raises(Exception, match="fragment owner threads.*are not contained in the active thread range"):
        tl.lower(_bound_fragment_kernel(True), target=TARGET, enable_device_compile=False)


@pytest.mark.parametrize("offset,replicas", [(0, 1), (64, 1), (64, 2)])
@pytest.mark.parametrize("exclude_owner", [False, True])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_bound_context_wider_than_fragment_owners(offset, replicas, exclude_owner):
    width = 64
    owner_extent = width * replicas
    layout = T.Fragment(
        (width,),
        replicate=replicas,
        forward_thread_fn=lambda i, rep=0: i + rep * width,
        forward_index_fn=lambda i: 0,
    )
    active_end = offset + owner_extent - int(exclude_owner)

    @T.prim_func
    def main(X: T.Tensor((width,), "float32"), Y: T.Tensor((width,), "float32")):
        with T.Kernel(1, threads=offset + owner_extent + 64):
            a = T.alloc_fragment((width,), "float32")
            tx = T.get_thread_binding()
            if tx >= offset:
                # The inferred binding includes 64 threads that own no data.
                for i in T.Parallel(width, loop_layout=layout):
                    a[i] = X[i]
            if tx >= offset:  # noqa: SIM102 - Exercise nested thread-domain refinement.
                if tx < active_end:
                    for i in T.Parallel(width):
                        a[i] = a[i] + 1
                    T.copy(a, Y, loop_layout=layout)

    if exclude_owner:
        # Updating only some owners would leave a logical element or replica
        # stale. A smaller execution context is legal only if it covers them.
        with pytest.raises(Exception, match="fragment owner threads.*are not contained in the active thread range"):
            tl.lower(main, target=TARGET, enable_device_compile=False)
        return

    kernel = tl.compile(main, out_idx=[1], target=TARGET)
    x = torch.arange(width, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(x), x + 1, atol=0, rtol=0)


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_reducer_with_inferred_consumer_loop():
    """A reducer update can derive its loop layout from the annotated input."""
    rows, cols = 2, 64
    layout = T.Fragment(
        (rows, cols),
        forward_thread_fn=lambda row, col: col,
        forward_index_fn=lambda row, col: row,
    )

    @T.prim_func
    def main(X: T.Tensor((rows, cols), "float32"), Y: T.Tensor((cols,), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_fragment((rows, cols), "float32")
            acc = T.alloc_reducer((cols,), "float32")
            result = T.alloc_fragment((cols,), "float32")
            T.annotate_layout({a: layout})
            tx = T.get_thread_binding()
            if tx >= 64:
                T.copy(X, a, loop_layout=layout)
                T.reducer_init(acc)
                for row, col in T.Parallel(rows, cols):
                    T.reducer_update(acc[col], a[row, col])
                T.finalize_reducer(acc, result)
                T.copy(result, Y)

    kernel = tl.compile(main, out_idx=[1], target=TARGET)
    x = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
    torch.testing.assert_close(kernel(x), x.sum(dim=0), atol=0, rtol=0)
