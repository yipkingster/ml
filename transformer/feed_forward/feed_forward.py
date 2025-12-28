from flax import linen as nn
import jax

class PositionwiseFeedForward(nn.Module):
    '''Feed Forward Network (FFN) to apply to individual seq after multi-head attention.

    Multi-head attention collects information from relationships between seqs.
    FFN focus on each seq independently and use non-linearity to infer the meaning/conclusion.
    This will create the higher-level feature.
    e.g. "Eiffle Towner is in..." as input sequences, "Eiffle" and "Tower" will show high similarity
    in attention output, but the FFN will turn up "Paris".
    
    Attributes:
        d_model: 512
        d_ff:    Usually 4x d_model = 2048
    '''
    d_model: int  # 512
    d_ff: int     # 2048

    @nn.compact
    def __call__(self, x):
        ''' Special callable method.

        Input:
            x: Output from multi-head attention mechanism: [batch, seq, d_model]
        Output:
            Linear tranformation of the input without changing the dimentionality.
        '''
        # TODO: Implement the Expansion -> Activation -> Compression logic
        # Hint: You need two nn.Dense layers and one activation function
        # Expand: We make 2048 from 512 so we can untangle the inner meaning of the seqs.
        # Also it helps more information to survive ReLU activation (ReLU kills data).
        dense1 = nn.Dense(self.d_ff)
        expanded = dense1(x)
        activated = jax.nn.relu(expanded)
        dense2 = nn.Dense(self.d_model)
        output = dense2(activated)
        return output