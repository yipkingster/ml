import os

# Set flag to simulate 8 devices
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=8"

# jax must be imported after setting the flag
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

def f(x, mesh):
    x = x+1
    # P() means "don't do any sharding, replicate everything to every shard".
    y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    return y

def main():
    # Verify the simulated devices work.
    print(f"Available device: {jax.device_count()}")
    # Works for jax cluster via jax.distributed.initialize()
    print(f"Devices: {jax.devices()}")
    # Works for devices that are locally attached.
    print(f"Local Devices: {jax.local_devices()}")
    # Check Jax version
    print(f"Jax Version: {jax.__version__}")

    # Put an array to 8 devices.
    # Creating hardware mesh.
    # The following way only works for Jax 0.4.35+
    # mesh = jax.make_mesh((4, 2), ("a", "b"))
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = Mesh(devices, axis_names=("a", "b"))

    # Initialize a 8192 x 8192 array following random normal distribution.
    x = jax.random.normal(jax.random.key(0), (8192, 8192))
    print("Single Device Sharding: ")
    jax.debug.visualize_array_sharding(x)
    y = jax.device_put(x, NamedSharding(mesh, P("a", "b")))
    print("8-device Sharding: ")
    jax.debug.visualize_array_sharding(y)
    print("After sin() computation:")

    # Do sin() and observe the sharding was respected:
    z = jnp.sin(y)
    jax.debug.visualize_array_sharding(z)

    # Do sharding on the (b, a) layout which means:
    # First dimension of x is on the 2 columns, and the 2nd dimension of x is on
    # the 4 rows:
    u = jax.device_put(x, NamedSharding(mesh, P("b", "a")))
    print("P(b, a) Layout: ")
    jax.debug.visualize_array_sharding(u)

    # Do sharding on the (a) only.
    v = jax.device_put(x, NamedSharding(mesh, P("a", None)))
    print("P(a, None) Layut: ")
    jax.debug.visualize_array_sharding(v)

    # Do sharding on (None, "a"), which means only shard the 2nd axis of x.
    w = jax.device_put(x, NamedSharding(mesh, P(None, "a")))
    print("P(None, a) Layout: ")
    jax.debug.visualize_array_sharding(w)

    # Do sharding on multiple/all mesh axes for a particular array dimension.
    # The 1st dimension will split through all devices while the 2nd dimension
    # is copied across all devices.
    r = jax.device_put(x, NamedSharding(mesh, P(("a", "b"), None)))
    print("P((a, b), None) Layout: ")
    jax.debug.visualize_array_sharding(r)

    # dot product will follow the mesh topo.
    s = jax.device_put(x, NamedSharding(mesh, P("a", None)))
    print("1st mx layout for dot product: ")
    jax.debug.visualize_array_sharding(s)
    t = jax.device_put(x, NamedSharding(mesh, P(None, "b")))
    print("2st mx layout for dot product: ")
    jax.debug.visualize_array_sharding(t)
    o = jnp.dot(s, t)
    print("dot product layout: ")
    jax.debug.visualize_array_sharding(o)

    # Replicate all output of a jit-ed function to every shard:
    p = f(x, mesh)
    print("Output of f() function in P() layout: ")
    jax.debug.visualize_array_sharding(p)

if __name__ == "__main__":
    main()