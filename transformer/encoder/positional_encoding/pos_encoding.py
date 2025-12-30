import jax.numpy as jnp
from flax import linen as nn

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000  # We pre-calculate up to this length

    def setup(self):
        self.pe = self.variable("const", "pe", self.init_pe)
        # No need to return anything. It will be ignored.

    def init_pe(self):
        # 1. Create a matrix of shape (max_len, d_model) filled with zeros
        # DELETE THIS LINE - Not needed since we are using stack now.
        pe = jnp.zeros((self.max_len, self.d_model))[:, jnp.newaxis]

        # 2. Create a vector 'position' containing [0, 1, 2, ..., max_len-1]
        position = jnp.arange(self.max_len)[:, jnp.newaxis]

        # 3. Create the 'div_term' (the denominator 10000^(...))
        # This is the tricky math part. 
        # Hint: exp(log(10000) * -2i / d_model) is numerically more stable.
        # Create initial array [0, 1, 2, ..., 256]
        power_vector = jnp.array(range(self.d_model//2))
        # Repeat each element twice so it becomes [0, 0, 1, 1, 2, 2, ..., 255, 255]
        # Alternative, use floor division: 
        # power_vector = jnp.arange(512) // 2
        # Since we are using stack, do not double the size.
        # power_vector = jnp.repeat(power_vector, 2)
        # Divide by d_model
        power_vector = power_vector / self.d_model 
        # To avoid overflow and underflow, compute in log space.
        div_term = jnp.exp(jnp.log(10000)*(-2)*power_vector)[jnp.newaxis, :]

        # Note that because of JNP vector broadcast, position of shape [5000, 1]
        # and div_term of shape [1, 512] are stretched to fit multiplication of
        # each other. The end result is of shape [5000, 512]
        # 4. Fill the even indices with sin()
        sin_value = jnp.sin(position * div_term)
        cos_value = jnp.cos(position * div_term)
        # This will add one more dimension and pair sin and cos at the deepest
        # level. The shape becomes [sen_len, d_model, 2]
        stacked = jnp.stack([sin_value, cos_value], axis=-1)
        # Flatten the final dimension so the [[Sin, Cos][Sin, Cos]] becomes
        # [Sin, Cos, Sin, Cos].
        stacked = stacked.reshape(self.max_len, self.d_model)

        # 6. Register this matrix as a constant (not a learnable weight!)
        # In Flax, we generally just store it as a variable with trainable=False
        # or just keep it as a constant if we don't need it in the checkpoint.
        # For this exercise, you can just return it in __call__ or store it.
        # Add Batch dimension: (1, max_len, d_model)
        return stacked[jnp.newaxis, :, :]

    def __call__(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        
        # Slice self.pe to match the actual length of x
        # If x has 50 words, we only need the first 50 rows of PE.
        pe_slice = self.pe.value[::, 0:x.shape[1], ::]
        
        # Add PE to x
        return x + pe_slice
