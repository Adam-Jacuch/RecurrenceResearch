from flax import nnx

from axiom import ax, Module, init, tensor

class Recurrence(Module):
    def __call__(self, x, alpha):
        def linear_combine(i, j):
            x_i, alpha_i = i
            x_j, alpha_j = j
            return (alpha_j * x_i + x_j), (alpha_j * alpha_i)
        return x[..., ax.sq.assoc_scan(fn=linear_combine, inputs=(alpha,)), ax.d]



class Step(Module):
    def __init__(self, step_idx: int = 0):
        self.step_idx = step_idx
        self.rec = Recurrence()

    def __call__(self, v, out, c, prev_fetched):
        c_norm = c[..., ax.d.norm_rms()]

        # Step 0 looks only at the token.
        # Step 1 looks at the token AND what Step 0 fetched.
        if self.step_idx == 0:
            gate_input = c_norm
        else:
            prev_norm = prev_fetched[..., ax.d.norm_rms()]
            gate_input = c_norm + prev_norm

        bias_init = init.linspace(-2.0, 6.5) if self.step_idx == 0 else init.linspace(0.0, 8.0)

        alphas = gate_input[..., ax.d.proj(bias_init=bias_init).sigmoid()]
        betas = gate_input[..., ax.d.proj().silu()]

        write_scale = (1.0 - alphas[..., ax.d.square()])[..., ax.d.clamp(min=1e-6).pow(0.5)]
        fetched = self.rec(v * betas * write_scale, alphas)

        out = out + fetched[..., ax.d.gate()]
        v = v + fetched[..., ax.d.gate(init_fn=init.zeros)]

        # Hand 'fetched' directly to the next step, no shadow stream needed
        return v, out, c, fetched

class Block(Module):
    def __init__(self, N: int) -> None:
        self.N = N
        self.steps = nnx.List(Step(step_idx=i) for i in range(N))

    def __call__(self, x):
        c = x[..., ax.d.conv(4, over=ax.sq.causal(), groups="depthwise").silu()]
        out_gate = x[..., ax.d.proj().silu()]

        v = out = c

        prev_fetched = c.zeros_like()

        for step in self.steps:
            v, out, c, prev_fetched = step(v, out, c, prev_fetched)

        out = out[..., ax.d.gate(tensor=out_gate)]
        return out[..., ax.d.norm_rms().proj(kernel_init=init.normal(1e-4)).silu()]

class Layer(Module):
    def __init__(self, dim: int, N: int, dropout : float) -> None:
        self.dropout = dropout
        self.dim = dim
        self.block = Block(N)
    def __call__(self, x):
        x = x + self.block(x)[..., ax.d.dropout(self.dropout)]
        return x + x[..., ax.d.norm_rms().proj(out=ax.d * 4).swiglu().proj(out=ax.d // 2, kernel_init=init.normal(1e-4)).dropout(self.dropout)]

class Model(Module):
    def __init__(self, vocab : int, dim : int, depth : int, N : int, dropout : float = 0.0) -> None:
        self.dropout = dropout
        self.vocab, self.dim = vocab, dim
        self.layers = nnx.List(Layer(dim, N, dropout) for _ in range(depth))
    def __call__(self, x, use_checkpointing=False):
        x, w = x.embed(vocab=ax.v(self.vocab), out=ax.d(self.dim), return_weight=True)
        x = x[..., ax.d.dropout(self.dropout)]
        for layer in self.layers:
            x = nnx.remat(layer)(x) if use_checkpointing else layer(x)
        return x[..., ax.d.proj(out=ax.v(self.vocab), weight=w, use_bias=False)]