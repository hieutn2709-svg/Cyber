"""Approved seed43/44 dev repeats on the frozen matched70 fold; no test mode."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from journal.scsp import matched_runtime as mr
from journal.scripts import train_gate_a as gate
from journal.scripts.train_matched_encoder import validate_output


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',choices=tuple(mr.PACKAGES),required=True)
    p.add_argument('--seed',type=int,choices=(43,44),required=True)
    for name in ('bundle','preflight','baseline-run','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p.add_argument('--resume',action='store_true')
    return p


def run(args):
    candidate=mr.seed_config(args.package,args.seed)
    args.mode='dev'
    validate_output(args)
    parent=mr.validate_preflight(args.preflight,mr.contract(args.bundle,args.package))
    mr.validate_cached_model(parent)
    mr.validate_seed_baseline(args.baseline_run,parent)
    from journal.scsp.training_recovery import run_lock
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with run_lock(args.output_dir):
        baseline_config,training,inventory=mr.config_paths(args.package)
        config=args.output_dir/baseline_config.name
        content=json.dumps(candidate,indent=2)+'\n'
        if args.resume:
            if not config.is_file() or config.read_text()!=content:
                raise ValueError('resume seed config mismatch')
        else:
            config.write_text(content)
        combined=gate._combined_config_hash([config,training,inventory])
        package=mr.seed_package(parent,args.seed,combined)
        evidence={'repeat_seed':args.seed,'parent_preflight_package_sha256':mr.digest(parent),
            'baseline_run_summary_sha256':gate._sha256_file(args.baseline_run/'run_summary.json'),
            'baseline_config_sha256':gate._sha256_file(baseline_config),
            'seed_config_sha256':gate._sha256_file(config),
            'engineering_gates_reused_from_seed42':True,'new_seed_preflight_claimed':False,
            'selection_scope':'baseline-cap128 validation only','test_evaluated':False}
        if args.resume and mr.read_json(args.output_dir/'seed_extension.json')!=evidence:
            raise ValueError('resume baseline evidence mismatch')
        if not args.resume:
            (args.output_dir/'seed_extension.json').write_text(json.dumps(evidence,indent=2)+'\n')
        delegated=argparse.Namespace(mode='dev',config=str(config),training_config=str(training),inventory=str(inventory),
            manifest=str(args.bundle/args.package/'split_manifest.json'),dataset=str(args.bundle/args.package/'dataset.json'),
            fold=1,seed=args.seed,output_dir=str(args.output_dir),device=args.device,overfit_epochs=20,
            encoder_package=package,epoch_recovery=True,resume=args.resume)
        if gate.mode_evaluates_test(delegated.mode):
            raise ValueError('test evaluation not authorized')
        return gate.run(delegated)


def main(argv=None):
    try:
        result=run(build_parser().parse_args(argv))
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}))
        return 2
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
