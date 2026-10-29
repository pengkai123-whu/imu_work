"""Parse every code cell; execute entry and GPU smoke cells locally in spk.

This deliberately skips dependency installation, full-cache preparation and full
training. It does not emulate or claim a connected Colab hosted runtime.
"""
import ast
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
notebook=json.loads((ROOT/'MEIO_YT_Colab.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
try:
    import nbformat
except ImportError:
    schema='nbformat not installed; structure and cell syntax checked'
else:
    nbformat.validate(nbformat.from_dict(notebook)); schema='nbformat validated'
env={'__name__':'__main__'}
def execute(index):
    code=''.join(notebook['cells'][index]['source'])
    exec(compile(code,f'notebook_cell_{index}','exec'),env)
execute(1)  # GPU detection, nvidia-smi
env.update(json=json,shutil=__import__('shutil'),CODE_DIR=ROOT)
execute(3)  # dataset location, run restoration
execute(4)  # configuration cell
execute(5)  # subprocess and Drive sync wrapper (local branch)
execute(7)  # unittest + real H5 GPU smoke through notebook wrapper
result=dict(passed=True,cell_count=len(notebook['cells']),schema=schema,
    executed_cells=[1,3,4,5,7],smoke_name=env['SMOKE_NAME'],
    note='local spk CUDA execution; Colab Drive/download/full training not executed')
target=ROOT/'runs/notebook_check.json'; target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
