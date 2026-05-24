import os, sys
p = '/groups/saalfeld/home/allierc/GraphData/config/cortex/cortex_matrix_Claude_00.yaml'
print('size:', os.path.getsize(p))
with open(p) as f:
    content = f.read()
print('len:', len(content))
print('first 3000 chars:')
print(content[:3000])
