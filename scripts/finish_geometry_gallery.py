import os,json
from pathlib import Path
from morqex.visualization.electrode_gallery import write_large_gallery
root=Path(os.environ['MORQEX_WORKSPACE'])/'atlas256'
write_large_gallery(root)
print(json.dumps({'html':str(root/'index.html'),'manifest':str(root/'manifest.json')}))
