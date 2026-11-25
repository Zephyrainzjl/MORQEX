"""Decision-stage error measurements; full evidence is rebuilt in the report."""
import json
from build_crossscale_report import WS,compare

def main():
    pairs={'LFP16':('closure_lfp_rom16_cycle','closure_lfp_exact_cycle')}
    pairs.update({f:(f'gem_{f}_rom8',f'gem_{f}_full') for f in ['isotropic','directional','CBD-rich']})
    pairs['directional_tolerance_first_cycle']=('gem_directional_full','gem_directional_full_tight_cycle')
    load=lambda n:json.loads((WS/'results/aging'/n/'result.json').read_text());result={}
    for name,(a,b) in pairs.items():
        result[name]=compare(load(a),load(b));print(name,json.dumps(result[name]),flush=True)
    (WS/'reports/crossscale_enrichment_accuracy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
