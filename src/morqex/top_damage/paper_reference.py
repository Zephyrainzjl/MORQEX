"""Traceable digitization of the published Ai et al. (2022) Figure 5.

Reads the author's Article.pdf; no PDF or author model is redistributed.
Vector vertices are plotted data, NOT original solver samples. Axis/line-width
uncertainty is retained and cannot certify a solver to machine precision.
"""
import hashlib
import json
from pathlib import Path

ARTICLE_SHA256 = '9dda9287cecb18836fd48498346d97b748ef5c3fc6e5231c525158d4190908cf'


def _multiply(a, b):
    return [a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5]]


def vector_paths(pdf, page_index):
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    reader = PdfReader(pdf)
    paths = []

    def walk(stream, resources, matrix, color):
        stack, points = [], []

        def point(x, y):
            return [matrix[0]*float(x)+matrix[2]*float(y)+matrix[4],
                    matrix[1]*float(x)+matrix[3]*float(y)+matrix[5]]

        for args, op in ContentStream(stream, reader).operations:
            if op == b'q': stack.append((matrix.copy(), color))
            elif op == b'Q': matrix, color = stack.pop()
            elif op == b'cm': matrix = _multiply(matrix, list(map(float, args)))
            elif op == b'RG': color = tuple(map(float, args))
            elif op == b'G': color = (float(args[0]),)*3
            elif op == b'm': points = [point(*args)]
            elif op == b'l': points.append(point(*args))
            elif op == b'c':
                controls = (points[-1], point(*args[:2]), point(*args[2:4]), point(*args[4:]))
                for t in (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.):
                    weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
                    points.append([sum(p[k]*w for p,w in zip(controls,weights)) for k in (0,1)])
            elif op in (b'S', b's', b'B', b'b'):
                if len(points)>1: paths.append(dict(color=color, points=points))
                points = []
            elif op == b'Do':
                obj = resources['/XObject'][args[0]].get_object()
                if obj.get('/Subtype') == '/Form':
                    walk(obj, obj.get('/Resources',resources),
                         _multiply(matrix,list(map(float,obj.get('/Matrix',[1,0,0,1,0,0])))), color)
        if stack: raise ValueError('Unbalanced PDF graphics state')

    page = reader.pages[page_index]
    walk(page.get_contents(), page['/Resources'], [1,0,0,1,0,0], (0,0,0))
    return paths


def extract_figure5(pdf, output):
    """Extract independently identified magenta 5C, R=5 um, a/R=.02 curves."""
    import numpy as np
    from .backend import require_local_budget
    require_local_budget()
    pdf, output = Path(pdf), Path(output)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != ARTICLE_SHA256:
        raise ValueError('Unrecognized PDF: re-audit figure coordinates before extraction')
    paths = vector_paths(pdf, 7)
    definitions = {
        'cracked_fraction': dict(x0=108.283,x300=215.339,y0=643.2,y1=722.484,v0=0.,v1=.0015),
        'mean_toughness_factor': dict(x0=390.118,x300=497.213,y0=643.2,y1=722.484,v0=.5,v1=1.1),
    }
    curves = {}
    for name, a in definitions.items():
        candidates = []
        for path in paths:
            p = np.asarray(path['points'])
            if (path['color']==(1.,0.,1.) and len(p)>100
                and abs(p[:,0].min()-a['x0'])<.05
                and abs(p[:,0].max()-a['x300'])<.05
                and p[:,1].min()>=a['y0']-.05 and p[:,1].max()<=a['y1']+.05):
                candidates.append(p)
        if len(candidates)!=1: raise ValueError(f'Ambiguous Figure 5 path: {name}')
        p = candidates[0]
        x = (p[:,0]-a['x0'])*300/(a['x300']-a['x0'])
        y = a['v0']+(p[:,1]-a['y0'])*(a['v1']-a['v0'])/(a['y1']-a['y0'])
        if np.any(np.diff(x)<-.05): raise ValueError('Curve reverses cycle axis')
        # PDF line thickness/read-off allowance; not a rigorous statistical CI.
        curves[name] = dict(cycle=x.tolist(),value=y.tolist(),axis_calibration=a,
            read_off_allowance=dict(pdf_points=.25,cycle=.25*300/(a['x300']-a['x0']),
                                   value=.25*(a['v1']-a['v0'])/(a['y1']-a['y0'])))
    payload = dict(schema_version=1,doi='10.1016/j.jpowsour.2022.231805',
        source='https://www.empaneda.com/wp-content/uploads/2022/12/COMSOLbatteryfatigue.zip',
        pdf_sha256=digest,page_index=7,figure='5(a,c)',
        kind='digitized published vector curves, not original numerical data',
        case=dict(geometry='sphere',radius_m=5e-6,c_rate=5.,crack_fraction=.02),curves=curves)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2))
    for name, curve in curves.items():
        np.savetxt(output.with_name(output.stem+'_'+name+'.csv'),
                   np.column_stack([curve['cycle'],curve['value']]),delimiter=',',
                   header='cycle,'+name,comments='')
    return payload
