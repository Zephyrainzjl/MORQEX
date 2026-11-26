"""Report only completed measured results, with the actual protocol in its title."""
import argparse,json,os
from pathlib import Path
from morqex.visualization.nodewise import build_node_report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',default='short');a=p.parse_args()
    ws=Path(os.environ['MORQEX_WORKSPACE']);source=ws/'results/sphere32_li'/a.case/'result.json'
    result=json.loads(source.read_text())
    if not result.get('completed'):raise RuntimeError('Requested protocol did not complete; do not publish it as completed')
    suffix='120 秒放电验证' if result['kind']=='short' else '完整循环协议'
    print(json.dumps(build_node_report(source,ws/f'reports/MORQEX_sphere32_Li_{a.case}.html',
        title='6 个正极节点 · 32³ 多颗粒 / Li · '+suffix)))
