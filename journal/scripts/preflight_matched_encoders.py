"""No-gradient compatibility gate for pinned matched encoder packages."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from journal.scsp import matched_runtime as mr, matched_windows as mw


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',choices=tuple(mr.PACKAGES),required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    return p


def run(args):
    if args.output_dir.exists():
        raise FileExistsError('preflight output already exists')
    expected=mr.contract(args.bundle,args.package)
    manifest=mr.read_json(args.bundle/args.package/'encoder_dataset_manifest.json')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    from journal.scripts import train_gate_a as gate
    from journal.scsp.config import GateAConfig
    from journal.scsp.training_config import GateATrainingConfig
    from journal.scsp.data import LabelInventory,load_clean_windows
    from journal.scsp.splits import load_fold_partition
    from journal.scsp.runtime_model import GateASpanPairModel
    gate._set_seed(42)
    device=gate._resolve_device(args.device)
    model_id,revision=mr.PACKAGES[args.package]
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,use_fast=True,trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('fast tokenizer required')
    backend=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if backend!=manifest['tokenizer']['resolved_backend_sha256']:
        raise ValueError('resolved tokenizer backend changed')
    tokens={'vocabulary_size':len(tokenizer),'special_token_ids':
        {key:getattr(tokenizer,key+'_token_id') for key in ('bos','pad','eos','unk')}}
    paths=mr.config_paths(args.package)
    config=GateAConfig.from_json(paths[0]);training=GateATrainingConfig.from_json(paths[1])
    inventory=LabelInventory.from_json(paths[2])
    windows=load_clean_windows(args.bundle/args.package/'dataset.json',inventory)
    partition=load_fold_partition(args.bundle/args.package/'split_manifest.json',1)
    train,_,_=gate._split_windows(windows,partition)
    width=gate._derive_width_cap(train,config.span_width_coverage)
    from transformers.utils.hub import cached_file
    resolved={'config.json':cached_file(model_id,'config.json',revision=revision)}
    weight_name=None
    for candidate in ('model.safetensors','pytorch_model.bin'):
        path=cached_file(model_id,candidate,revision=revision,_raise_exceptions_for_missing_entries=False)
        if path is not None:
            resolved[candidate]=path;weight_name=candidate;break
    if weight_name is None:
        raise ValueError('pinned model weights unavailable')
    for path in resolved.values():
        mr.validate_snapshot_path(path,revision)
    resolved_hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in resolved.items()}
    encoder=AutoModel.from_pretrained(model_id,revision=revision,trust_remote_code=False,
        use_safetensors=weight_name.endswith('.safetensors'))
    model=GateASpanPairModel(encoder=encoder,hidden_size=encoder.config.hidden_size,
        num_entity_classes=len(inventory.trainable_entity_types)+1,
        num_relation_types=len(inventory.relation_types),max_width=width,
        width_embedding_dim=training.width_embedding_dim,context_dim=training.context_dim,
        distance_embedding_dim=training.distance_embedding_dim,max_distance=config.max_relation_token_distance)
    model.to(device);model.eval()
    length=max(len(w.input_ids) for w in windows)
    # Synthetic input; no train/validation/test annotation or prediction is accessed.
    ids=torch.full((1,length),tokens['special_token_ids']['unk'],dtype=torch.long,device=device)
    ids[0,0]=tokens['special_token_ids']['bos'];ids[0,-1]=tokens['special_token_ids']['eos']
    with torch.no_grad():
        states=model.encode(ids,torch.ones_like(ids))
    mr.validate_components(encoder.config,encoder.get_input_embeddings().num_embeddings,tokens,states.shape,length)
    if not torch.isfinite(states).all():
        raise ValueError('non-finite encoder output')
    enc=list(model.encoder_parameters());heads=list(model.head_parameters())
    if not enc or not heads or {id(x) for x in enc}&{id(x) for x in heads}:
        raise ValueError('optimizer parameter groups empty or overlapping')
    if {id(x) for x in enc+heads}!={id(x) for x in model.parameters()}:
        raise ValueError('optimizer parameter groups incomplete')
    package={'contract':expected,'resolved_model_files_sha256':resolved_hashes,'model_class':type(encoder).__name__,'tokenizer_class':type(tokenizer).__name__,
        'tokenizer_backend_sha256':backend,**tokens,'hidden_size':encoder.config.hidden_size,
        'max_positions':encoder.config.max_position_embeddings,'width_cap':width,
        'encoder_parameter_count':sum(x.numel() for x in enc),'head_parameter_count':sum(x.numel() for x in heads),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime':{**gate._environment(device),'transformers_version':transformers.__version__},
        'upstream_url':f'https://huggingface.co/{model_id}/tree/{revision}'}
    report={'status':'passed','test_evaluated':False,'gradient_steps':0,'input_length':length,
        'last_hidden_state_shape':list(states.shape),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'encoder_package_sha256':mr.digest(package)}
    if not report['deterministic_algorithms']:
        raise ValueError('deterministic runtime required')
    mw.write_atomic_bundle(args.output_dir,{'encoder_package.json':package,'runtime_compatibility.json':report})
    return report


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
