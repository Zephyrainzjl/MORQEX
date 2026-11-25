"""Keep README/API in source; preserve research documents in the workspace."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent/'morph_workspace'

def main():
    archive=WORKSPACE/'research_archive/pre_open_voxel';archive.mkdir(parents=True,exist_ok=True);records=[]
    for path in sorted(ROOT.rglob('*.md')):
        if '.git' in path.parts or path.name in {'README.md','API.md'}:continue
        relative=path.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=path.read_bytes():raise ValueError(f'Archive would overwrite different content: {target}')
        shutil.copy2(path,target);records.append({'source':str(relative),'archive':str(target),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.name=='THIRD_PARTY_NOTICES.md':path.rename(path.with_suffix('.txt'))
        else:path.unlink()
    upstream=ROOT/'provenance/gemsim_upstream/README.md'
    if upstream.exists():upstream.rename(upstream.with_suffix('.txt'))
    if records:(archive/'manifest.json').write_text(json.dumps(records,indent=2))
    api=ROOT/'API.md';prefix=api.read_text().split('<!-- GENERATED_API_INDEX -->')[0]+'<!-- GENERATED_API_INDEX -->\n';sections=[]
    for path in sorted((ROOT/'src/morqex').rglob('*.py')):
        tree=ast.parse(path.read_text());entries=[]
        for item in tree.body:
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not item.name.startswith('_'):
                args=item.args if hasattr(item,'args') else next((n.args for n in item.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
                signature=ast.unparse(args) if args is not None else '';doc=(ast.get_docstring(item) or '').split('\n')[0].strip()
                entries.append(f'- `{item.name}({signature})`'+(f' — {doc}' if doc else ''))
                if isinstance(item,ast.ClassDef):
                    for method in item.body:
                        if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and not method.name.startswith('_'):
                            signature=ast.unparse(method.args);description=(ast.get_docstring(method) or '').split('\n')[0].strip()
                            entries.append(f'- `{item.name}.{method.name}({signature})`'+(f' — {description}' if description else ''))
        if entries:
            rel=path.relative_to(ROOT);module='.'.join(path.relative_to(ROOT/'src').with_suffix('').parts)
            sections.append(f'### [{module}]({rel})\n\n'+'\n'.join(entries))
    api.write_text(prefix+'\n\n'.join(sections)+'\n');print(json.dumps({'archived_documents':len(records),'api_modules':len(sections),'archive':str(archive)},indent=2))
if __name__=='__main__':main()
