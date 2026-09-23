# SecureBERT v1 Dataset Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic tokenizer-neutral reconstruction pipeline that proves RoBERTa parity before it is allowed to produce a SecureBERT v1 clean-window dataset.

**Architecture:** Parse the 52 Label Studio documents into immutable character-coordinate records, load the frozen clean-window file only as the logical-window and parity contract, and align annotations through tokenizer offsets. A pure core accepts a small tokenizer protocol so unit tests do not download models; the CLI lazily loads pinned Hugging Face tokenizers and writes deterministic datasets, manifests, and aggregate audits.

**Tech Stack:** Python 3.10+, standard library, `transformers` fast tokenizers at CLI runtime, existing `journal.scsp` inventory/data adapters, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-17-securebert-v1-comparison-design.md`

## Global Constraints

- Source corpus identity is exactly 52 documents with stable `id` order.
- Frozen controlled dataset SHA-256 is `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`.
- RoBERTa package is `FacebookAI/roberta-base` at revision `c2a5e573587885ce23744cf330ee7c402f0df16f`.
- SecureBERT v1 package is `ehsanaghaei/SecureBERT` at revision `3a47918dd874e5c769efd152b51c5756a953fb67`.
- Model and tokenizer revisions are immutable 40-character hexadecimal commit IDs.
- The RoBERTa parity contract is 52 documents, 67 windows, 1,353 primary entities, 28 auxiliary entities, 564 relations, and 561 core-to-core relations.
- Entity type `file_paths` in the Label Studio source is canonically normalized to inventory type `file-paths`; no other undeclared label rewrite is allowed.
- Empty, discontinuous, ambiguous, special-token-only, and truncated entity alignments fail closed with document and entity identifiers.
- Exclusions required to reproduce the frozen dataset are explicit audit records; no entity or relation is silently dropped.
- Dataset construction never reads split metrics and never evaluates model predictions.
- Raw CTI text and entity surface text are never written to committed audit or provenance artifacts.
- No training, `dev`, `full`, or test evaluation is authorized by this plan.

## Review Focus

- Unicode normalization or mojibake repair changes character offsets: the loader must preserve source coordinates or fail before tokenization; Task 1 tests this.
- Fast-tokenizer offsets may be relative to an isolated logical-window slice: Task 3 tests conversion back to document character coordinates.
- An entity may touch a boundary without being wholly contained: Task 3 tests explicit `crosses_window_boundary` exclusion rather than clipping.
- A relation may survive only at one endpoint: Task 4 tests explicit orphan-relation audit and forbids serialized dangling endpoints.
- Canonical digests can accidentally become order-insensitive: Task 2 tests that document, window, entity, and relation order mutations change the digest.

---

### Task 1: Tokenizer-neutral source records and Label Studio loader

**Files:**
- Create: `journal/scsp/encoder_dataset.py`
- Create: `tests/test_scsp_encoder_dataset.py`

**Interfaces:**
- Consumes: Label Studio JSON with `data.text`, one accepted annotation result list, `labels` entities, and labeled `relation` records.
- Produces: `SourceEntity`, `SourceRelation`, `SourceDocument`, `normalize_source_text(text)`, and `load_label_studio_documents(path)`.

- [ ] **Step 1: Write failing source-loader tests**

```python
class EncoderDatasetSourceTests(unittest.TestCase):
    def test_loader_preserves_order_coordinates_and_normalizes_one_legacy_label(self):
        rows = [label_studio_row(
            doc_id=420,
            text="A malware.exe",
            entities=[("e1", 2, 13, "file_paths")],
            relations=[],
        )]
        docs = load_fixture(rows)
        self.assertEqual(docs[0].doc_seq_index, 0)
        self.assertEqual(docs[0].doc_id, "420")
        self.assertEqual(docs[0].entities[0].key, ("e1", 2, 13, "file-paths"))

    def test_loader_rejects_text_repair_that_invalidates_source_slice(self):
        rows = [label_studio_row(
            doc_id=420,
            text="cafÃ©",
            entities=[("e1", 0, 5, "malware")],
            relations=[],
        )]
        with self.assertRaisesRegex(ValueError, "420.*e1.*coordinates"):
            load_fixture(rows)

    def test_loader_rejects_duplicate_entity_and_unknown_relation_endpoint(self):
        rows = [label_studio_row(
            doc_id=420,
            text="alpha beta",
            entities=[("e1", 0, 5, "malware"), ("e1", 6, 10, "tool")],
            relations=[("e1", "missing", "uses")],
        )]
        with self.assertRaisesRegex(ValueError, "duplicate entity_id"):
            load_fixture(rows)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderDatasetSourceTests -v`

Expected: import failure because `journal.scsp.encoder_dataset` does not exist.

- [ ] **Step 3: Implement immutable source records and strict parsing**

```python
@dataclass(frozen=True, slots=True)
class SourceEntity:
    entity_id: str
    char_start: int
    char_end: int
    label: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.entity_id, self.char_start, self.char_end, self.label)

@dataclass(frozen=True, slots=True)
class SourceRelation:
    source_id: str
    target_id: str
    label: str

@dataclass(frozen=True, slots=True)
class SourceDocument:
    doc_seq_index: int
    doc_id: str
    text: str
    entities: tuple[SourceEntity, ...]
    relations: tuple[SourceRelation, ...]

def normalize_source_text(text: str) -> str:
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired = text
    return repaired
```

Validate root/list shape, unique document IDs, exactly one non-empty accepted annotation, half-open character spans, non-empty surface slices, unique entity IDs, non-empty relation labels, and both endpoints. If normalization changes length or any annotated source slice, raise a coordinate error instead of guessing.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderDatasetSourceTests -v`

Expected: all source-loader tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add journal/scsp/encoder_dataset.py tests/test_scsp_encoder_dataset.py
git commit -m "feat: load tokenizer-neutral CTI annotations"
```

### Task 2: Frozen window contract and canonical semantic digest

**Files:**
- Modify: `journal/scsp/encoder_dataset.py`
- Modify: `tests/test_scsp_encoder_dataset.py`

**Interfaces:**
- Consumes: frozen clean-window JSON and `LabelInventory`.
- Produces: `LogicalWindow`, `load_logical_window_contract(path)`, `canonical_window_payload(rows)`, `semantic_sha256(rows)`, and `compare_reconstruction(expected, actual)`.

- [ ] **Step 1: Write failing contract and digest tests**

```python
class FrozenWindowContractTests(unittest.TestCase):
    def test_contract_retains_only_identity_and_global_boundaries(self):
        contract = load_contract([clean_window_row()])
        self.assertEqual(
            contract,
            (LogicalWindow(0, "420", 0, 0, 3),),
        )

    def test_semantic_digest_is_deterministic_and_order_sensitive(self):
        first = [clean_window_row(doc_id="420"), clean_window_row(doc_id="400")]
        same = json.loads(json.dumps(first))
        reversed_rows = list(reversed(first))
        self.assertEqual(semantic_sha256(first), semantic_sha256(same))
        self.assertNotEqual(semantic_sha256(first), semantic_sha256(reversed_rows))

    def test_parity_reports_first_nested_mismatch(self):
        expected = [clean_window_row()]
        actual = json.loads(json.dumps(expected))
        actual[0]["entity_spans"][0]["token_end"] += 1
        with self.assertRaisesRegex(
            ValueError,
            r"row 0.*entity_spans\[0\]\.token_end",
        ):
            compare_reconstruction(expected, actual)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.FrozenWindowContractTests -v`

Expected: missing symbols for the contract/digest API.

- [ ] **Step 3: Implement strict contract parsing and canonical hashing**

```python
@dataclass(frozen=True, slots=True)
class LogicalWindow:
    doc_seq_index: int
    doc_id: str
    window_index: int
    token_start_global: int
    token_end_global: int

def semantic_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        canonical_window_payload(rows),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

The canonical payload keeps ordered identity fields, vectors, entity IDs/types/character and token coordinates, and relation endpoints/types. It intentionally removes entity surface `text` only; no list is sorted. `compare_reconstruction` walks mappings and lists in order and reports the first JSON-style path plus expected/actual values.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.FrozenWindowContractTests -v`

Expected: all contract and digest tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add journal/scsp/encoder_dataset.py tests/test_scsp_encoder_dataset.py
git commit -m "feat: define frozen encoder window contract"
```

### Task 3: Offset alignment and logical-window character projection

**Files:**
- Modify: `journal/scsp/encoder_dataset.py`
- Modify: `tests/test_scsp_encoder_dataset.py`

**Interfaces:**
- Consumes: tokenizer offset mappings as half-open `(start, end)` pairs, special-token masks, `SourceEntity`, and `LogicalWindow`.
- Produces: `AlignedEntity`, `AlignmentExclusion`, `WindowProjection`, `align_entity(...)`, and `project_logical_window(...)`.

- [ ] **Step 1: Write failing alignment tests**

```python
class EncoderAlignmentTests(unittest.TestCase):
    def test_alignment_uses_all_overlapping_non_special_subwords(self):
        entity = SourceEntity("e1", 2, 8, "malware")
        aligned = align_entity(
            doc_id="420",
            window_index=0,
            entity=entity,
            offsets=((0, 0), (0, 4), (4, 8), (0, 0)),
            special_tokens_mask=(1, 0, 0, 1),
            window_char_start=0,
            window_char_end=8,
        )
        self.assertEqual((aligned.token_start, aligned.token_end), (1, 2))

    def test_alignment_rejects_special_only_and_discontinuous_tokens(self):
        entity = SourceEntity("e1", 2, 8, "malware")
        with self.assertRaisesRegex(ValueError, "420.*e1.*empty"):
            align_entity(
                doc_id="420", window_index=0, entity=entity,
                offsets=((0, 0),), special_tokens_mask=(1,),
                window_char_start=0, window_char_end=8,
            )
        with self.assertRaisesRegex(ValueError, "420.*e1.*discontinuous"):
            align_entity(
                doc_id="420", window_index=0, entity=entity,
                offsets=((0, 4), (20, 21), (4, 8)),
                special_tokens_mask=(0, 0, 0),
                window_char_start=0, window_char_end=21,
            )

    def test_boundary_crossing_entity_is_audited_not_clipped(self):
        projection = project_logical_window(
            document=source_document(entity=(8, 14)),
            logical_window=LogicalWindow(0, "420", 0, 0, 3),
            document_offsets=((0, 4), (4, 8), (8, 12)),
        )
        self.assertEqual(projection.exclusions[0].reason, "crosses_window_boundary")
        self.assertEqual(projection.entities, ())
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderAlignmentTests -v`

Expected: missing alignment/projection symbols.

- [ ] **Step 3: Implement fail-closed overlap alignment**

```python
@dataclass(frozen=True, slots=True)
class AlignedEntity:
    entity_id: str
    label: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int

@dataclass(frozen=True, slots=True)
class AlignmentExclusion:
    doc_id: str
    window_index: int
    entity_id: str
    reason: str

@dataclass(frozen=True, slots=True)
class WindowProjection:
    char_start: int
    char_end: int
    entities: tuple[SourceEntity, ...]
    exclusions: tuple[AlignmentExclusion, ...]

def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)
```

Require matching offset/mask lengths, contiguous selected indices, full character coverage from the first to last selected token, and a wholly contained entity. Convert window-relative offsets back to document coordinates before overlap checks.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderAlignmentTests -v`

Expected: all alignment tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add journal/scsp/encoder_dataset.py tests/test_scsp_encoder_dataset.py
git commit -m "feat: align character entities to encoder tokens"
```

### Task 4: Deterministic clean-window builder and exclusion accounting

**Files:**
- Modify: `journal/scsp/encoder_dataset.py`
- Modify: `tests/test_scsp_encoder_dataset.py`

**Interfaces:**
- Consumes: `SourceDocument`, `LogicalWindow`, `LabelInventory`, and a callable tokenizer returning complete `input_ids`, `attention_mask`, `offset_mapping`, and `special_tokens_mask` arrays.
- Produces: `BuildResult(rows, exclusions, audit)` and `build_encoder_windows(...)`.

- [ ] **Step 1: Write failing builder tests using a deterministic fake tokenizer**

```python
class EncoderWindowBuilderTests(unittest.TestCase):
    def test_builder_serializes_local_spans_and_only_complete_relations(self):
        result = build_encoder_windows(
            documents=(source_document_with_two_entities_and_relation(),),
            logical_windows=(LogicalWindow(0, "420", 0, 0, 3),),
            tokenizer=WhitespaceOffsetTokenizer(),
            inventory=inventory(),
            encoder_id="unit/encoder",
            encoder_revision="a" * 40,
            max_length=8,
        )
        row = result.rows[0]
        self.assertEqual(row["label_mask"], [False, True, True, True, False])
        self.assertEqual(
            [(item["entity_id"], item["token_start"], item["token_end"])
             for item in row["entity_spans"]],
            [("e1", 1, 1), ("e2", 3, 3)],
        )
        self.assertEqual(row["relations"], [
            {"source_id": "e1", "target_id": "e2", "type": "uses"}
        ])

    def test_builder_records_orphan_relation_instead_of_serializing_it(self):
        result = build_encoder_windows(
            documents=(source_document_with_boundary_crossing_target(),),
            logical_windows=(LogicalWindow(0, "420", 0, 0, 3),),
            tokenizer=WhitespaceOffsetTokenizer(),
            inventory=inventory(),
            encoder_id="unit/encoder",
            encoder_revision="a" * 40,
            max_length=8,
        )
        self.assertEqual(result.rows[0]["relations"], [])
        self.assertEqual(result.audit["relation_exclusions_by_reason"], {
            "endpoint_not_in_window": 1
        })

    def test_builder_requires_protocol_amendment_instead_of_truncating(self):
        with self.assertRaisesRegex(
            ProtocolAmendmentRequired,
            "420.*window 0.*matched subwindow",
        ) as caught:
            build_encoder_windows(
                documents=(long_source_document(),),
                logical_windows=(LogicalWindow(0, "420", 0, 0, 510),),
                tokenizer=TruncatingTokenizer(),
                inventory=inventory(),
                encoder_id="unit/encoder",
                encoder_revision="a" * 40,
                max_length=512,
            )
        self.assertGreater(len(caught.exception.subwindows), 1)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderWindowBuilderTests -v`

Expected: missing builder API.

- [ ] **Step 3: Implement deterministic row construction**

```python
@dataclass(frozen=True, slots=True)
class BuildResult:
    rows: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]
    audit: dict[str, Any]

class ProtocolAmendmentRequired(ValueError):
    def __init__(self, message: str, subwindows: Sequence[tuple[int, int]]):
        super().__init__(message)
        self.subwindows = tuple(subwindows)

def build_encoder_windows(
    *,
    documents: Sequence[SourceDocument],
    logical_windows: Sequence[LogicalWindow],
    tokenizer: TokenizerProtocol,
    inventory: LabelInventory,
    encoder_id: str,
    encoder_revision: str,
    max_length: int,
) -> BuildResult:
    documents_by_key = {
        (document.doc_seq_index, document.doc_id): document
        for document in documents
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for logical_window in logical_windows:
        document = documents_by_key[
            (logical_window.doc_seq_index, logical_window.doc_id)
        ]
        projection = project_logical_window(
            document=document,
            logical_window=logical_window,
            document_offsets=tokenizer.document_offsets(document.text),
        )
        encoding = tokenizer.encode_window(
            document.text[projection.char_start:projection.char_end],
            max_length=max_length,
        )
        if encoding.truncated or len(encoding.input_ids) > max_length:
            subwindows = deterministic_whitespace_subwindows(
                document.text,
                projection.char_start,
                projection.char_end,
                tokenizer,
                max_length,
            )
            raise ProtocolAmendmentRequired(
                f"{document.doc_id} window {logical_window.window_index} "
                "requires a matched subwindow protocol amendment",
                subwindows,
            )
        row, row_exclusions = materialize_clean_window_row(
            document=document,
            logical_window=logical_window,
            projection=projection,
            encoding=encoding,
            inventory=inventory,
        )
        rows.append(row)
        exclusions.extend(row_exclusions)
    audit = audit_encoder_build(rows, exclusions, inventory)
    return BuildResult(tuple(rows), tuple(exclusions), audit)
```

Add the exact `TokenizerProtocol`, `WindowEncoding`, and `WindowProjection`
records used above, plus `materialize_clean_window_row` and
`audit_encoder_build`. The implementation builds `bieos_labels`,
`role_labels`, and `core_label_mask` with the same semantics as the frozen
format; orders entities and relations exactly as their source annotation
order; rejects duplicate aligned spans; and verifies every emitted relation
endpoint exists in the same row.

The audit contains source/emitted/excluded entity and relation counts, reason
counters, document/window counts, label counts, total subword count, subwords
per non-whitespace character, per-document window counts, entity subword-length
distribution, boundary-crossing count, unknown-token count/rate, truncation and
overflow counts, and entity/relation preservation counts. Exclusion rows
contain IDs, coordinates, labels, and reasons but no raw text.

If SecureBERT tokenization would overflow a frozen logical window, compute and
record deterministic whitespace-boundary character subwindows in the audit,
but return status `protocol_amendment_required` and write no dataset. The same
character subwindows would have to be applied to both RoBERTa and SecureBERT;
that matched re-baseline requires separate design approval and is not an
implicit fallback in this plan.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset.EncoderWindowBuilderTests -v`

Expected: all builder tests pass.

- [ ] **Step 5: Run the existing clean-window adapter tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_data_adapter -v`

Expected: existing adapter behavior remains green.

- [ ] **Step 6: Commit Task 4**

```bash
git add journal/scsp/encoder_dataset.py tests/test_scsp_encoder_dataset.py
git commit -m "feat: build deterministic encoder windows"
```

### Task 5: Dataset manifest, parity gate, and atomic artifact writer

**Files:**
- Create: `journal/scsp/encoder_dataset_artifacts.py`
- Create: `tests/test_scsp_encoder_dataset_artifacts.py`

**Interfaces:**
- Consumes: `BuildResult`, source/frozen file paths, tokenizer metadata, expected frozen rows, and output directory.
- Produces: `build_encoder_dataset_manifest(...)`, `validate_roberta_parity(...)`, and `write_encoder_dataset_bundle(...)`.

- [ ] **Step 1: Write failing manifest and atomicity tests**

```python
class EncoderDatasetArtifactTests(unittest.TestCase):
    def test_roberta_parity_requires_exact_frozen_counts_and_semantics(self):
        manifest = build_manifest_fixture(
            document_count=52, window_count=67,
            primary_entity_count=1353, auxiliary_entity_count=28,
            relation_count=564, core_to_core_relation_count=561,
        )
        validate_roberta_parity(manifest)
        with self.assertRaisesRegex(ValueError, "relation_count"):
            validate_roberta_parity({**manifest, "relation_count": 563})

    def test_writer_leaves_no_partial_bundle_when_parity_fails(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "bundle"
            with self.assertRaisesRegex(ValueError, "semantic parity"):
                write_encoder_dataset_bundle(
                    output_dir=output,
                    rows=(clean_window_row(),),
                    manifest=bad_parity_manifest(),
                    audit={},
                    exclusions=(),
                    expected_rows=(different_clean_window_row(),),
                    require_roberta_parity=True,
                )
            self.assertFalse(output.exists())
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset_artifacts -v`

Expected: artifact module import failure.

- [ ] **Step 3: Implement deterministic manifest and atomic directory replace**

```python
def build_encoder_dataset_manifest(
    *, source_sha256: str, frozen_dataset_sha256: str,
    encoder_id: str, encoder_revision: str,
    tokenizer_class: str, tokenizer_files: Mapping[str, str],
    build_result: BuildResult, dataset_sha256: str,
    semantic_sha256_value: str, builder_git_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "frozen_dataset_sha256": frozen_dataset_sha256,
        "encoder": {"id": encoder_id, "revision": encoder_revision},
        "tokenizer": {
            "id": encoder_id,
            "revision": encoder_revision,
            "class": tokenizer_class,
            "files_sha256": dict(sorted(tokenizer_files.items())),
        },
        "dataset_sha256": dataset_sha256,
        "semantic_sha256": semantic_sha256_value,
        "builder_git_commit": builder_git_commit,
        **build_result.audit,
    }
```

Write `dataset.json`, `encoder_dataset_manifest.json`, `tokenizer_audit.json`, and `alignment_exclusions.json` into a sibling temporary directory; fsync/close files; rename only after every parity and hash check passes. Serialize JSON with UTF-8, sorted mapping keys, two-space indentation, and one trailing newline.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset_artifacts -v`

Expected: all artifact tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add journal/scsp/encoder_dataset_artifacts.py tests/test_scsp_encoder_dataset_artifacts.py
git commit -m "feat: enforce encoder dataset parity artifacts"
```

### Task 6: Guarded reconstruction CLI with lazy tokenizer loading

**Files:**
- Create: `journal/scripts/build_encoder_comparison_dataset.py`
- Create: `tests/test_build_encoder_comparison_dataset_cli.py`

**Interfaces:**
- Consumes: source annotations, frozen dataset, label inventory, mode `roberta-parity` or `securebert`, explicit output directory, and immutable package IDs/revisions.
- Produces: an executable CLI and a verified artifact bundle; SecureBERT mode requires the path and SHA-256 of a passed RoBERTa parity manifest.

- [ ] **Step 1: Write failing CLI/help and guard tests**

```python
class BuildEncoderComparisonDatasetCliTests(unittest.TestCase):
    def test_help_does_not_import_transformers(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("roberta-parity", result.stdout)
        self.assertIn("securebert", result.stdout)

    def test_mutable_revision_is_rejected_before_tokenizer_load(self):
        with self.assertRaisesRegex(ValueError, "immutable 40-character"):
            validate_cli_contract(
                mode="securebert",
                model_id="ehsanaghaei/SecureBERT",
                revision="main",
                roberta_parity_manifest=valid_parity_manifest_path(),
            )

    def test_securebert_requires_verified_roberta_parity_manifest(self):
        with self.assertRaisesRegex(ValueError, "RoBERTa parity"):
            validate_cli_contract(
                mode="securebert",
                model_id="ehsanaghaei/SecureBERT",
                revision="3a47918dd874e5c769efd152b51c5756a953fb67",
                roberta_parity_manifest=None,
            )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_build_encoder_comparison_dataset_cli -v`

Expected: CLI module/file is missing.

- [ ] **Step 3: Implement parser, preflight contract, and lazy tokenizer factory**

```python
def load_fast_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise ValueError("encoder dataset construction requires a fast tokenizer")
    return tokenizer
```

`roberta-parity` hardcodes the approved RoBERTa package and requires frozen dataset SHA equality before reading its rows. `securebert` hardcodes the approved SecureBERT package and verifies the supplied parity manifest records `parity_status == "passed"`, the frozen SHA, the approved RoBERTa revision, and its own file SHA. The CLI never accepts `full`, fold, threshold, checkpoint, or prediction arguments.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_build_encoder_comparison_dataset_cli -v`

Expected: all CLI tests pass without importing `transformers` for `--help`.

- [ ] **Step 5: Compile all new Python files**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m py_compile journal/scsp/encoder_dataset.py journal/scsp/encoder_dataset_artifacts.py journal/scripts/build_encoder_comparison_dataset.py tests/test_scsp_encoder_dataset.py tests/test_scsp_encoder_dataset_artifacts.py tests/test_build_encoder_comparison_dataset_cli.py`

Expected: exit code 0 and no output.

- [ ] **Step 6: Commit Task 6**

```bash
git add journal/scripts/build_encoder_comparison_dataset.py tests/test_build_encoder_comparison_dataset_cli.py
git commit -m "feat: add guarded encoder dataset builder"
```

### Task 7: Dataset-gate regression verification and documentation

**Files:**
- Create: `journal/README_SECUREBERT_V1.md`
- Modify only if required by verified behavior: files created in Tasks 1–6

**Interfaces:**
- Consumes: all dataset-gate modules and tests.
- Produces: operator commands for synthetic verification and later Colab parity execution; no real-data result is claimed or committed in this task.

- [ ] **Step 1: Write the operator documentation**

Document the two-phase command order, immutable revisions, expected files, expected frozen counts, failure semantics, external output placement, and the explicit statement: `This implementation step does not authorize training, --mode dev, --mode full, or test evaluation.`

- [ ] **Step 2: Run all new dataset-gate tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_dataset tests.test_scsp_encoder_dataset_artifacts tests.test_build_encoder_comparison_dataset_cli -v`

Expected: all dataset-gate tests pass.

- [ ] **Step 3: Run existing pure-Python boundary suites**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_data_adapter tests.test_scsp_gate_a_config_splits tests.test_scsp_gate_a_runner tests.test_reproducibility -v`

Expected: all selected existing tests pass.

- [ ] **Step 4: Run the full repository suite in the provisioned PyTorch environment**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`

Expected: 0 failures and 0 errors. If `torch` is absent, record the environment blocker and do not claim full-suite success; rerun this exact command in the existing Colab/PyTorch environment before PR preparation.

- [ ] **Step 5: Verify scope and protected files**

Run: `git diff --check origin/journal/scsp-q2-securebert-v1-design...HEAD && git diff --name-only origin/journal/scsp-q2-securebert-v1-design...HEAD`

Expected: no whitespace errors; changes are limited to this plan, dataset-gate modules/tests, and SecureBERT documentation. No Gate A/B/C decoder or evaluation file changes.

- [ ] **Step 6: Commit documentation and final verified adjustments**

```bash
git add journal/README_SECUREBERT_V1.md docs/superpowers/plans/2026-09-23-securebert-v1-dataset-gate.md
git commit -m "docs: document SecureBERT dataset gate"
```

## Completion boundary

This plan ends when the reconstruction code and synthetic verification are complete. Running the real 52-document RoBERTa parity audit is a separate controlled execution step because it requires the external frozen dataset and pinned tokenizer runtime. SecureBERT dataset generation is forbidden until that audit produces a passed manifest. Runtime/training integration begins only under `docs/superpowers/plans/2026-09-23-securebert-v1-runtime-training.md`.
