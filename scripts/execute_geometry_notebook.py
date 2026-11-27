"""Execute a clean copy with this bounded job's Python kernel and workspace."""
import json,os,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(os.environ['MORQEX_WORKSPACE']);out=root/'notebook_example';out.mkdir(exist_ok=True)
kernel=root/'jupyter/kernels/morqex-gallery';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
 'display_name':'MORQEX geometry','language':'python'}))
os.environ['JUPYTER_PATH']=str(root/'jupyter')
source=Path(__file__).resolve().parents[1]/'tutorials/notebooks/geometry_workflow.ipynb'
notebook=nbformat.read(source,as_version=4)
client=NotebookClient(notebook,timeout=240,kernel_name='morqex-gallery',resources={'metadata':{'path':str(out)}})
client.execute()
target=out/'geometry_workflow.executed.ipynb';nbformat.write(notebook,target)
errors=[output for c in notebook.cells if c.cell_type=='code' for output in c.get('outputs',[]) if output.output_type=='error']
if errors:raise ArithmeticError('Notebook contains errors')
print(json.dumps({'executed_notebook':str(target),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),'errors':0}))
