"""Compare the default sampler to the immutable pre-ablation implementation.

Run as a module from the repository root. This is an engineering audit, with
synthetic candidate lists only; it does not train or score a pretrained model.
"""
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import types
from journal.scsp.candidates import build_training_pairs
from journal.scsp.pairs import GoldRelation
from journal.scsp.structures import GoldSpan, SpanCandidate

BASE='63a31ba67089ea2bb7bf0016ff91f9f0fa4e0921'


def main():
    root=Path(__file__).resolve().parents[2]
    source=subprocess.check_output(['git','show',BASE+':journal/scsp/candidates.py'],cwd=root)
    # Fixed trusted repository source, never a user-supplied module path.
    module=types.ModuleType('journal.scsp._frozen_candidates')
    module.__package__='journal.scsp'
    sys.modules[module.__name__]=module
    exec(compile(source,'frozen candidates at '+BASE,'exec'),module.__dict__)
    cases=0; repairs=0; unaffected=0
    for documents,types_per_boundary,cutoff,ratio,seed in itertools.product(
            (1,2),(1,2,3),(0,2,96),(0,1,3,100),(0,1,2,42,43,44,101)):
        spans=tuple(SpanCandidate('d'+str(d),pos,pos,label='T'+str(t))
                    for d in range(documents) for pos in (0,3,150) for t in range(types_per_boundary))
        gold=(GoldRelation(GoldSpan('d0',0,0,'T0'),GoldSpan('d0',3,3,'T0'),'r'),
              GoldRelation(GoldSpan('d0',150,150,'T0'),GoldSpan('d0',0,0,'T0'),'r'),
              GoldRelation(GoldSpan('d0',3,3,'T0'),GoldSpan('d0',3,3,'T0'),'r'))
        expected_stats={};actual_stats={}
        expected=module.build_training_pairs(spans,gold,cutoff,ratio,seed=seed,diagnostics=expected_stats)
        actual=build_training_pairs(spans,gold,cutoff,ratio,seed=seed,diagnostics=actual_stats)
        assert asdict(actual)==asdict(expected)
        assert actual_stats==expected_stats
        repaired_stats={}
        repaired=build_training_pairs(spans,gold,cutoff,ratio,seed=seed,
            diagnostics=repaired_stats,negative_policy='boundary_replace')
        assert repaired_stats['sampled']['collision_groups']==0
        assert repaired_stats['eligible']==expected_stats['eligible']
        assert repaired_stats['negative_policy']['baseline_sampled_batch_sha256']==expected_stats['sampled_batch_sha256']
        if not expected_stats['sampled']['collision_groups']:
            assert repaired==actual
            unaffected+=1
        else:repairs+=1
        cases+=1
    result={'scope':'synthetic sampler compatibility audit; no pretrained training or scoring',
        'baseline_commit':BASE,'baseline_source_sha256':hashlib.sha256(source).hexdigest(),
        'current_source_sha256':hashlib.sha256((root/'journal/scsp/candidates.py').read_bytes()).hexdigest(),
        'cases':cases,'default_batches_and_diagnostics_exact':True,
        'intervention_batches_without_conflicts':cases,'unaffected_batches_exact':unaffected,
        'batches_requiring_repair':repairs}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
