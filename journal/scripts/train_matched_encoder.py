"""Guarded matched-corpus overfit, smoke and approved baseline dev execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from journal.scsp import matched_runtime as mr
from journal.scripts import train_gate_a as gate


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',choices=tuple(mr.PACKAGES),required=True)
    p.add_argument('--mode',choices=('overfit','smoke','dev'),required=True)
    for name in ('bundle','preflight','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    p.add_argument('--overfit-run',type=Path)
    p.add_argument('--smoke-run',type=Path)
    p.add_argument('--resume',action='store_true',help='resume an incomplete run from its atomic epoch recovery state')
    return p


def validate_output(args):
    if getattr(args,'resume',False):
        if args.mode == 'overfit':
            raise ValueError('overfit recovery is not supported')
        if (args.output_dir/'run_summary.json').exists():
            raise ValueError('run is already complete')
        if not (args.output_dir/'recovery.pt').is_file():
            raise ValueError('resume requires complete epoch recovery state')
    elif args.output_dir.exists():
        raise FileExistsError('run output already exists')


def run(args):
    mr.validate_mode(args.package,args.mode)
    validate_output(args)
    expected=mr.contract(args.bundle,args.package)
    package=mr.validate_preflight(args.preflight,expected)
    mr.validate_cached_model(package)
    mr.validate_previous_gate(args.mode,args.overfit_run,args.smoke_run,package)
    config,training,inventory=mr.config_paths(args.package)
    delegated=argparse.Namespace(mode=args.mode,config=str(config),training_config=str(training),
        inventory=str(inventory),manifest=str(args.bundle/args.package/'split_manifest.json'),
        dataset=str(args.bundle/args.package/'dataset.json'),fold=1,seed=42,
        output_dir=str(args.output_dir),device=args.device,overfit_epochs=20,encoder_package=package,
        epoch_recovery=args.mode != 'overfit',resume=getattr(args,'resume',False))
    if gate.mode_evaluates_test(delegated.mode):
        raise ValueError('test evaluation not authorized')
    return gate.run(delegated)


def main(argv=None):
    try:
        result=run(build_parser().parse_args(argv))
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
