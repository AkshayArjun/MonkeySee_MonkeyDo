# test_loss.py — run this with: python3 test_loss.py
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import sys
sys.path.insert(0, '/home/focas/mcdepth_ws/src/robot_retarget/robot_retarget')
import rx200_kinematics as rx_kine

x0     = np.zeros(5, dtype=np.float64)
target = np.array([0.0, 0.0, 0.3,
                   0.2, 0.0, 0.2,
                   0.35, 0.0, 0.1,
                   0.0, 0.0, 0.0, 1.0], dtype=np.float64)

def loss_fn(x):
    val, grad = rx_kine.loss_and_grad_fn(
        jnp.array(x, dtype=jnp.float32),
        jnp.array(target, dtype=jnp.float32),
        jnp.array([0.67, 0.33], dtype=jnp.float32)
    )
    val_np  = float(val)
    grad_np = np.array(grad, dtype=np.float64)
    grad_np = np.where(np.isfinite(grad_np), grad_np, 0.0)
    print(f"val={val_np:.4f}  grad={grad_np}  grad_dtype={grad_np.dtype}")
    return val_np, grad_np

res = minimize(
    fun=loss_fn,
    x0=x0,
    method='SLSQP',
    jac=True,
    bounds=[(-3.14,3.14),(-1.88,1.97),(-1.88,1.62),(-1.74,2.14),(-3.14,3.14)],
    options={'maxiter': 50, 'ftol': 1e-4}
)

print(f"\nResult: {res.message}")
print(f"x: {res.x}")