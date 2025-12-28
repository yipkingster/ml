from flax.linen import nn

class PositionwiseFeedForward(nn.Module):
    d_model: int  # 512
    d_ff: int     # 2048

    @nn.compact
    def __call__(self, x):
        # TODO: Implement the Expansion -> Activation -> Compression logic
        # Hint: You need two nn.Dense layers and one activation function
        return x