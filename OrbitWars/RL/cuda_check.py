"""cuda_check.py — confirm PyTorch sees the RTX 4060 and that OrbitNet actually
runs on it. Run:  python cuda_check.py

If it prints 'CUDA available: False', you have the CPU-only torch build.
Fix (NVIDIA, CUDA 12.x):
  pip uninstall torch -y
  pip install torch --index-url https://download.pytorch.org/whl/cu121
then re-run this.
"""
import time, torch

print("torch version     :", torch.__version__)
print("CUDA available    :", torch.cuda.is_available())
print("CUDA build (torch):", torch.version.cuda)

if not torch.cuda.is_available():
    print("\n>>> CUDA NOT available — you're on the CPU-only torch build.")
    print(">>> Install the CUDA build (see comment at top of this file), then re-run.")
    raise SystemExit(0)

dev = "cuda"
print("device name       :", torch.cuda.get_device_name(0))
print("capability        :", torch.cuda.get_device_capability(0))

# --- 1) raw tensor op on GPU ---
x = torch.randn(4096, 4096, device=dev)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
print(f"\n[matmul] 10x 4096^2 matmul on GPU OK  ({time.time()-t0:.3f}s)")

# --- 2) the ACTUAL model on GPU: forward + backward ---
try:
    from model import OrbitNet, build_edge, N_MAX, NODE_F
    B, N = 32, N_MAX
    net = OrbitNet().to(dev)
    nf = torch.randn(B, N, NODE_F, device=dev)
    nm = (torch.rand(B, N, device=dev) > 0.2).float()
    am = (torch.rand(B, N, N, device=dev) > 0.5).float()
    edge = build_edge(nf)
    tl, fl, v = net(nf, edge, nm, am)
    loss = tl.float().nan_to_num(neginf=0).sum() + fl.sum() + v.sum()
    loss.backward()
    print(f"[OrbitNet] forward+backward on GPU OK  "
          f"(tgt {tuple(tl.shape)}, frac {tuple(fl.shape)}, val {tuple(v.shape)})")
    print("\nAll good — your scripts will use the GPU automatically (they already")
    print("select 'cuda' when available). No code changes needed.")
except Exception as e:
    print("\n[OrbitNet] FAILED on GPU:", repr(e))
    print(">>> torch sees CUDA but the model failed — paste this output and we'll debug.")
