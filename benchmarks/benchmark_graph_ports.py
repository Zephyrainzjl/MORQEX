"""Matched frozen-network terminal-only cost against a reused sparse factor."""
import json,os,time
import numpy as np
from morqex.top_damage.graph_transport import voxel_network
from morqex.workspace import Workspace


def main():
    rng=np.random.default_rng(104);k=np.exp(rng.normal(size=(16,16,16)))
    tick=time.perf_counter();graph=voxel_network(k,(.5e-6,)*3);op=graph.homogenize()['operator'];offline=time.perf_counter()-tick
    q=rng.normal(size=(500,2));outputs=[];timings=[]
    # Give the full reference the same assembly/factor reuse as the reduced
    # path; sparse block extraction is not charged to individual queries.
    Lbb=op.L[op.terminals][:,op.terminals].tocsr();Lbi=op.Lib.T.tocsr()
    for reduced in (False,True):
        start=time.perf_counter();answer=[]
        for v in q:
            if reduced:answer.append(op.currents(v))
            else:
                # Efficient full reference: reuse SAME LU; do not reassemble.
                u=op.factor.solve(-op.Lib@v/op.scale)
                answer.append(Lbb@v+Lbi@u)
        timings.append(time.perf_counter()-start);outputs.append(np.asarray(answer))
    error=float(np.max(abs(outputs[0]-outputs[1]))/np.max(abs(outputs[0])))
    if error>1e-10:raise ArithmeticError('Matched terminal response mismatch')
    record={'vertices':graph.network.vertex_count,'queries':len(q),'offline_s':offline,
        'full_reused_factor_s':timings[0],'schur_s':timings[1],'speedup':timings[0]/timings[1],
        'relative_max_error':error,'scope':'Frozen passive graph, terminal currents only; no full-field output, nonlinear reactions or fatigue speedup claimed'}
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);ws.path('reports/graph_port_benchmark.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
