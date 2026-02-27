# test_jax_grad.py
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/focas/mcdepth_ws/src/robot_retarget/robot_retarget')
import rx200_kinematics as rx_kine

target  = jnp.array([0.0, 0.0, 0.3,
                     0.2, 0.0, 0.2,
                     0.35, 0.0, 0.1,
                     0.0, 0.0, 0.0, 1.0])
weights = jnp.array([0.67, 0.33])
q0      = jnp.zeros(5)

# Direct JAX grad — not through value_and_grad
grad_fn = jax.grad(rx_kine.ocra_loss)
g = grad_fn(q0, target, weights)
print(f"Direct JAX grad:        {g}")

# Through value_and_grad as scipy would call it
val, g2 = rx_kine.loss_and_grad_fn(q0, target, weights)
print(f"value_and_grad:         val={val:.4f}  grad={g2}")

# With float32 as your node uses
val3, g3 = rx_kine.loss_and_grad_fn(
    jnp.array(q0, dtype=jnp.float32),
    jnp.array(target, dtype=jnp.float32),
    jnp.array(weights, dtype=jnp.float32)
)
print(f"value_and_grad float32: val={val3:.4f}  grad={g3}")