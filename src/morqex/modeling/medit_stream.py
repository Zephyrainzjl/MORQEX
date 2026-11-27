"""Bounded ASCII Medit exchange, including volume and interface references."""
import itertools
import numpy as np


def write_and_verify_medit(path,points,cells,domains,facets,markers):
    with path.open('x') as out:
        out.write('MeshVersionFormatted 2\nDimension 3\nVertices\n'+str(len(points))+'\n')
        for start in range(0,len(points),50000):
            p=points[start:start+50000]
            np.savetxt(out,np.column_stack((p,np.zeros(len(p),np.int32))),fmt=['%.17g','%.17g','%.17g','%d'])
        out.write('Tetrahedra\n'+str(len(cells))+'\n')
        for start in range(0,len(cells),50000):
            c=cells[start:start+50000].astype(np.int64)+1
            np.savetxt(out,np.column_stack((c,domains[start:start+50000])),fmt='%d')
        if len(facets):
            out.write('Triangles\n'+str(len(facets))+'\n')
            for start in range(0,len(facets),50000):
                c=facets[start:start+50000].astype(np.int64)+1
                np.savetxt(out,np.column_stack((c,markers[start:start+50000])),fmt='%d')
        out.write('End\n')
    with path.open() as stream:
        def line():
            value=stream.readline()
            if not value:raise ArithmeticError('Truncated Medit file')
            return value.strip()
        if line()!='MeshVersionFormatted 2' or line()!='Dimension 3':raise ArithmeticError('Invalid Medit version/dimension')
        def block(name,count,width,dtype,compare):
            if line()!=name or int(line())!=count:raise ArithmeticError('Medit block schema mismatch: '+name)
            for start in range(0,count,5000):
                stop=min(start+5000,count);rows=list(itertools.islice(stream,stop-start))
                if len(rows)!=stop-start:raise ArithmeticError('Truncated Medit data')
                values=np.fromstring(' '.join(rows),sep=' ',dtype=dtype).reshape(stop-start,width)
                compare(start,stop,values)
        def vertices(start,stop,values):
            np.testing.assert_allclose(values[:,:3],points[start:stop],rtol=1e-13,atol=1e-20)
            np.testing.assert_array_equal(values[:,3],np.zeros(stop-start,np.int32))
        def tetrahedra(start,stop,values):
            np.testing.assert_array_equal(values[:,:4]-1,cells[start:stop]);np.testing.assert_array_equal(values[:,4],domains[start:stop])
        def triangles(start,stop,values):
            np.testing.assert_array_equal(values[:,:3]-1,facets[start:stop]);np.testing.assert_array_equal(values[:,3],markers[start:stop])
        block('Vertices',len(points),4,np.float64,vertices)
        block('Tetrahedra',len(cells),5,np.int64,tetrahedra)
        if len(facets):block('Triangles',len(facets),4,np.int64,triangles)
        if line()!='End' or stream.read().strip():raise ArithmeticError('Invalid Medit terminator/trailing content')
    return {'readback':'independent bounded Medit parser; all coordinates/cells/domain references/interface references',
        'domain_readback_passed':True,'boundary_readback_passed':True}
