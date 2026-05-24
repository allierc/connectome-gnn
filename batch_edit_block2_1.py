import os
for slot in range(20):
    path = f"/groups/saalfeld/home/allierc/GraphData/config/cortex/cortex_unique_matrix_Claude_{slot:02d}.yaml"
    with open(path) as f:
        text = f.read()
    text = text.replace("  lr: 0.002\n", "  lr: 0.001\n")
    old_sched = (
        "  - 0.002\n"
        "  - 0.002\n"
        "  - 0.001\n"
        "  - 0.001\n"
        "  - 0.0004\n"
        "  - 0.0004\n"
        "  - 0.0002\n"
        "  - 0.0002\n"
        "  - 0.0002\n"
        "  - 0.0002\n"
    )
    new_sched = (
        "  - 0.001\n"
        "  - 0.001\n"
        "  - 0.0005\n"
        "  - 0.0005\n"
        "  - 0.0002\n"
        "  - 0.0002\n"
        "  - 0.0001\n"
        "  - 0.0001\n"
        "  - 0.0001\n"
        "  - 0.0001\n"
    )
    count = text.count(old_sched)
    text = text.replace(old_sched, new_sched)
    text = text.replace("lr peak\\\n  \\ 2e-3", "lr peak\\\n  \\ 1e-3 Block2.1")
    with open(path, "w") as f:
        f.write(text)
    print(f"slot {slot:02d}: schedules replaced={count}")
