import torch, time

# In RNN parlance, the dimensions we care about are:
# # of time-steps (T)
# Batch size (B)
# Hidden size/number of "channels" (C)
T, B, C = 50, 50, 1024


# A module that defines a single "bidirectional LSTM". This is simply two
# LSTMs applied to the same sequence, but one in reverse
class BidirectionalRecurrentLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_f = torch.nn.LSTM(input_size=C, hidden_size=C)
        self.cell_b = torch.nn.LSTM(input_size=C, hidden_size=C)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Forward layer
    #     output_f, _ = self.cell_f(x)

    #     # Backward layer. Flip input in the time dimension (dim 0), apply the
    #     # layer, then flip the outputs in the time dimension
    #     x_rev = torch.flip(x, dims=[0])
    #     output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
    #     output_b_rev = torch.flip(output_b, dims=[0])

    #     return torch.cat((output_f, output_b_rev), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward layer - fork() so this can run in parallel to the backward
        # layer
        future_f = torch.jit.fork(self.cell_f, x)

        # Backward layer. Flip input in the time dimension (dim 0), apply the
        # layer, then flip the outputs in the time dimension
        x_rev = torch.flip(x, dims=[0])
        output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
        output_b_rev = torch.flip(output_b, dims=[0])

        # Retrieve the output from the forward layer. Note this needs to happen
        # *after* the stuff we want to parallelize with
        output_f, _ = torch.jit.wait(future_f)

        return torch.cat((output_f, output_b_rev), dim=2)


# An "ensemble" of `BidirectionalRecurrentLSTM` modules. The modules in the
# ensemble are run one-by-one on the same input then their results are
# stacked and summed together, returning the combined result.
class LSTMEnsemble(torch.nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.n_models = n_models
        self.models = torch.nn.ModuleList(
            [BidirectionalRecurrentLSTM() for _ in range(self.n_models)]
        )

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     results = []
    #     for model in self.models:
    #         results.append(model(x))
    #     return torch.stack(results).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        futures = [torch.jit.fork(model, x) for model in self.models]
        results = [torch.jit.wait(fut) for fut in futures]
        return torch.stack(results).sum(dim=0)


# For a head-to-head comparison to what we're going to do with fork/wait, let's
# instantiate the model and compile it with TorchScript
ens = torch.jit.script(LSTMEnsemble(n_models=4))

# Normally you would pull this input out of an embedding table, but for the
# purpose of this demo let's just use random data.
x = torch.rand(T, B, C)

# Let's run the model once to warm up things like the memory allocator
ens(x)

x = torch.rand(T, B, C)

# Let's see how fast it runs!
s = time.time()
with torch.autograd.profiler.profile() as prof:
    ens(x)
prof.export_chrome_trace("parallel.json")
print("Inference took", time.time() - s, " seconds")
