import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for name, fn in (("cv2", lambda: __import__("cv2")),
                 ("davis", lambda: __import__("connectome_gnn.generators.davis",
                                              fromlist=["AugmentedVideoDataset"]))):
    try:
        m = fn(); print(f"{name} OK", getattr(m, "__version__", ""))
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {e}")
