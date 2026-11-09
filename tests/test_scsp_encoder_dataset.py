import copy
import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp import encoder_dataset as ed


def source_row(doc_id='420', text='alpha uses beta', entities=None, relations=None):
    if entities is None:
        entities = [('e1', 0, 5, 'malware'), ('e2', 11, 15, 'tool')]
    if relations is None:
        relations = [('e1', 'e2', 'uses')]
    return {'id': doc_id, 'data': {'text': text}, 'annotations': [{'result': [
        {'id': eid, 'type': 'labels', 'value': {'start': a, 'end': b,
         'text': text[a:b], 'labels': [label]}} for eid, a, b, label in entities
    ] + [{'type': 'relation', 'from_id': a, 'to_id': b, 'labels': [label],
          'direction': 'right'} for a, b, label in relations]}]}


def load_source(rows):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'source.json'
        p.write_text(json.dumps(rows), encoding='utf-8')
        return ed.load_label_studio_documents(p)


class EncoderDatasetSourceTests(unittest.TestCase):
    def test_loader_preserves_order_coordinates_and_normalizes_one_legacy_label(self):
        docs = load_source([source_row(text='A malware.exe',
            entities=[('e1', 2, 13, 'file_paths')], relations=[])])
        self.assertEqual((docs[0].doc_seq_index, docs[0].doc_id), (0, '420'))
        self.assertEqual(docs[0].entities[0].key, ('e1', 2, 13, 'file-paths'))
        self.assertEqual(docs[0].text, 'A malware.exe')

    def test_loader_rejects_text_repair_that_invalidates_source_slice(self):
        with self.assertRaisesRegex(ValueError, '420.*e1.*coordinates'):
            load_source([source_row(text='cafÃ©', entities=[('e1', 0, 5, 'malware')], relations=[])])

    def test_loader_rejects_duplicate_entity_and_unknown_relation_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'duplicate entity_id'):
            load_source([source_row(entities=[('e1', 0, 5, 'malware'), ('e1', 11, 15, 'tool')])])
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            load_source([source_row(relations=[('e1', 'missing', 'uses')])])

    def test_loader_rejects_ambiguous_annotations_and_invalid_spans(self):
        row = source_row(); row['annotations'] *= 2
        with self.assertRaisesRegex(ValueError, 'annotation'):
            load_source([row])
        for span in [(-1, 3), (4, 4), (0, 99), (True, 3)]:
            with self.subTest(span=span), self.assertRaises(ValueError):
                load_source([source_row(entities=[('e1', *span, 'malware')], relations=[])])

    def test_loader_rejects_duplicate_docs_surface_mismatch_and_unknown_result(self):
        with self.assertRaisesRegex(ValueError, 'document'):
            load_source([source_row(), source_row()])
        row = source_row(); row['annotations'][0]['result'][0]['value']['text'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            load_source([row])
        row = source_row(); row['annotations'][0]['result'].append({'type': 'choices'})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            load_source([row])


def clean_row():
    return {'doc_seq_index': 0, 'doc_id': '420', 'window_index': 0,
        'token_start_global': 0, 'token_end_global': 3,
        'input_ids': [0, 10, 11, 12, 2], 'attention_mask': [1]*5,
        'label_mask': [False, True, True, True, False],
        'bieos_labels': ['O', 'S-malware', 'O', 'S-tool', 'O'],
        'role_labels': ['O', 'ROLE_1', 'O', 'ROLE_2', 'O'],
        'core_label_mask': [False, True, True, True, False],
        'entity_spans': [{'entity_id': 'e1', 'type': 'malware', 'role': 'ROLE_1',
                         'token_start': 1, 'token_end': 1, 'char_start': 0, 'char_end': 5, 'text': 'alpha'},
                        {'entity_id': 'e2', 'type': 'tool', 'role': 'ROLE_2',
                         'token_start': 3, 'token_end': 3, 'char_start': 11, 'char_end': 15, 'text': 'beta'}],
        'relations': [{'source_id': 'e1', 'target_id': 'e2', 'type': 'uses'}]}


class FrozenWindowContractTests(unittest.TestCase):
    def test_contract_retains_only_identity_and_global_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'frozen.json'; p.write_text(json.dumps([clean_row()]))
            self.assertEqual(ed.load_logical_window_contract(p), (ed.LogicalWindow(0, '420', 0, 0, 3),))
            p.write_text(json.dumps([clean_row(), clean_row()]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ed.load_logical_window_contract(p)

    def test_semantic_digest_preserves_all_list_orders_and_ignores_only_surface(self):
        rows = [clean_row(), {**clean_row(), 'doc_id': '421', 'doc_seq_index': 1}]
        changed = copy.deepcopy(rows); changed[0]['entity_spans'][0]['text'] = 'ignored'
        self.assertEqual(ed.semantic_sha256(rows), ed.semantic_sha256(changed))
        mutations = [list(reversed(rows))]
        for key in ('input_ids', 'entity_spans'):
            changed = copy.deepcopy(rows); changed[0][key].reverse(); mutations.append(changed)
        changed = copy.deepcopy(rows); changed[0]['relations'].append({'source_id':'e2','target_id':'e1','type':'uses'})
        reverse = copy.deepcopy(changed); reverse[0]['relations'].reverse()
        self.assertNotEqual(ed.semantic_sha256(changed), ed.semantic_sha256(reverse))
        for changed in mutations:
            self.assertNotEqual(ed.semantic_sha256(rows), ed.semantic_sha256(changed))

    def test_parity_reports_first_nested_mismatch_without_surface_text(self):
        expected = [clean_row()]; actual = copy.deepcopy(expected)
        actual[0]['entity_spans'][0]['token_end'] += 1
        with self.assertRaisesRegex(ValueError, r'row 0.*entity_spans\[0\]\.token_end'):
            ed.compare_reconstruction(expected, actual)
        actual = copy.deepcopy(expected); actual[0]['extra'] = 'private'
        with self.assertRaisesRegex(ValueError, 'keys'):
            ed.compare_reconstruction(expected, actual)


class EncoderAlignmentTests(unittest.TestCase):
    def test_source_whitespace_boundaries_keep_original_character_coordinates(self):
        entity = ed.SourceEntity('space', 0, 7, 'malware')
        got = self.align(((0, 0), (1, 6), (0, 0)), (1, 0, 1), entity=entity,
            window_char_start=1, window_char_end=6, document_text=' alpha ')
        self.assertEqual((got.char_start, got.char_end, got.token_start, got.token_end),
                         (0, 7, 1, 1))

    def test_uncovered_non_whitespace_still_fails_with_source_context(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.align(((0, 0), (1, 5), (0, 0)), (1, 0, 1),
                entity=ed.SourceEntity('partial', 0, 5, 'malware'),
                window_char_start=0, window_char_end=5, document_text='alpha')

    def test_projection_keeps_annotation_with_only_whitespace_outside_window(self):
        entity = ed.SourceEntity('space', 0, 7, 'malware')
        doc = ed.SourceDocument(0, '420', ' alpha ', (entity,), ())
        got = ed.project_logical_window(document=doc,
            logical_window=ed.LogicalWindow(0, '420', 0, 0, 1),
            document_offsets=((1, 6),))
        self.assertEqual(got.entities, (entity,))
        self.assertEqual(got.exclusions, ())

    def align(self, offsets, mask, entity=None, **kwargs):
        return ed.align_entity(doc_id='420', window_index=0,
            entity=entity or ed.SourceEntity('e1', 2, 8, 'malware'),
            offsets=offsets, special_tokens_mask=mask,
            window_char_start=kwargs.pop('window_char_start', 0),
            window_char_end=kwargs.pop('window_char_end', 30), **kwargs)

    def test_alignment_uses_all_overlapping_non_special_subwords(self):
        got = self.align(((0, 0), (0, 4), (4, 8), (0, 0)), (1, 0, 0, 1))
        self.assertEqual((got.token_start, got.token_end), (1, 2))

    def test_alignment_rejects_empty_discontinuous_partial_and_ambiguous(self):
        cases = [(((0, 0),), (1,), 'empty'),
                 (((0, 4), (20, 21), (4, 8)), (0, 0, 0), 'discontinuous'),
                 (((3, 8),), (0,), 'truncated'),
                 (((0, 5), (4, 8)), (0, 0), 'ambiguous')]
        for offsets, mask, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, '420.*e1.*'+error):
                self.align(offsets, mask)

    def test_boundary_crossing_entity_is_audited_not_clipped(self):
        doc = ed.SourceDocument(0, '420', 'alpha beta long',
            (ed.SourceEntity('e1', 8, 14, 'malware'),), ())
        got = ed.project_logical_window(document=doc,
            logical_window=ed.LogicalWindow(0, '420', 0, 0, 3),
            document_offsets=((0,4), (4,8), (8,12)))
        self.assertEqual(got.exclusions[0].reason, 'crosses_window_boundary')
        self.assertEqual(got.entities, ())

    def test_window_relative_offsets_are_projected_to_document_coordinates(self):
        got = self.align(((0,0),(0,5),(0,0)), (1,0,1),
            entity=ed.SourceEntity('e2',11,15,'tool'),
            window_char_start=10, window_char_end=15, offsets_are_local=True)
        self.assertEqual((got.char_start, got.char_end, got.token_start), (11,15,1))


class SourceCompatibilityTests(unittest.TestCase):
    def test_frozen_endpoint_convention_preserves_from_to_for_left_ui_direction(self):
        row = source_row(); row['annotations'][0]['result'][-1]['direction'] = 'left'
        relation = load_source([row])[0].relations[0]
        self.assertEqual((relation.source_id, relation.target_id, relation.label), ('e1','e2','uses'))


class WhitespaceTokenizer:
    unk_token_id = 99
    def document_offsets(self, text):
        import re
        return tuple((m.start(),m.end()) for m in re.finditer(r'\S+', text))
    def encode_window(self, text, *, max_length):
        offsets = self.document_offsets(text)
        return ed.WindowEncoding((0, *range(10,10+len(offsets)), 2), (1,)*(len(offsets)+2),
            ((0,0), *offsets, (0,0)), (1, *((0,)*len(offsets)), 1), False)


def inventory():
    from journal.scsp.data import LabelInventory
    return LabelInventory(('malware','tool'), ('file-paths',), ('uses',))


class EncoderWindowBuilderTests(unittest.TestCase):
    def test_frozen_overlap_policy_excludes_all_conflicting_reference_members_with_audit(self):
        docs = load_source([source_row(entities=[('e1',0,5,'malware'),
            ('overlap',0,10,'tool'),('e2',11,15,'tool')])])
        got = self.build(docs=docs, restore_frozen_overlap_policy=True)
        self.assertEqual([e['entity_id'] for e in got.rows[0]['entity_spans']], ['e2'])
        self.assertEqual(got.rows[0]['relations'], [])
        self.assertEqual(got.rows[0]['bieos_labels'], ['O','O','O','S-tool','O'])
        self.assertEqual(got.rows[0]['entity_spans'][0]['role'], 'ROLE_2')
        excluded = [e for e in got.exclusions if e['reason']=='frozen_reference_overlap']
        self.assertEqual([(e['entity_id'],e['char_start'],e['char_end'],e['type'])
                         for e in excluded], [('e1',0,5,'malware'),('overlap',0,10,'tool')])
        self.assertEqual(got.audit['source_entity_count'], 3)
        self.assertEqual(got.audit['reference_overlap_excluded_count'], 2)

    def test_target_tokenization_cannot_select_reference_conflicts(self):
        class Target(WhitespaceTokenizer):
            def document_offsets(self, text):
                raise AssertionError('reference policy must not use target offsets')
            def encode_window(self, text, *, max_length):
                return WhitespaceTokenizer().encode_window(text, max_length=max_length)
        docs=load_source([source_row(entities=[('e1',0,5,'malware'),
            ('overlap',0,10,'tool'),('e2',11,15,'tool')])])
        got=self.build(docs=docs,tokenizer=Target(),reference_tokenizer=WhitespaceTokenizer(),
                       restore_frozen_overlap_policy=True)
        self.assertEqual([e['entity_id'] for e in got.rows[0]['entity_spans']], ['e2'])

    def build(self, docs=None, windows=None, tokenizer=None, **kwargs):
        return ed.build_encoder_windows(documents=docs or load_source([source_row()]),
            logical_windows=windows or (ed.LogicalWindow(0,'420',0,0,3),),
            tokenizer=tokenizer or WhitespaceTokenizer(), inventory=inventory(),
            encoder_id='unit/encoder', encoder_revision='a'*40,
            max_length=kwargs.pop('max_length',8), **kwargs)

    def test_builder_serializes_local_spans_and_complete_relations(self):
        result = self.build()
        ed.compare_reconstruction([clean_row()], result.rows)
        self.assertEqual(result.audit['relation_count'],1)
        self.assertEqual(result.audit['primary_entity_count'],2)

    def test_builder_records_orphan_relation_instead_of_serializing_it(self):
        docs = load_source([source_row(entities=[('e1',0,5,'malware'),('e2',8,15,'tool')])])
        result = self.build(docs=docs,windows=(ed.LogicalWindow(0,'420',0,0,2),))
        self.assertEqual(result.rows[0]['relations'],[])
        self.assertEqual(result.audit['relation_exclusions_by_reason'],{'endpoint_not_in_window':1})
        self.assertEqual(result.audit['boundary_crossing_count'],1)

    def test_builder_requires_amendment_instead_of_truncating(self):
        with self.assertRaisesRegex(ed.ProtocolAmendmentRequired, 'matched subwindow') as caught:
            self.build(max_length=4)
        self.assertEqual(caught.exception.subwindows, ((0,11),(11,15)))

    def test_reference_tokenizer_controls_character_windows_and_offsets_are_local(self):
        class Target(WhitespaceTokenizer):
            def document_offsets(self,text):
                raise AssertionError('must use reference tokenizer for global boundaries')
            def encode_window(self,text,*,max_length):
                return WhitespaceTokenizer().encode_window(text,max_length=max_length)
        got = self.build(windows=(ed.LogicalWindow(0,'420',1,2,3),),
            tokenizer=Target(), reference_tokenizer=WhitespaceTokenizer())
        self.assertEqual(got.rows[0]['entity_spans'][0]['entity_id'],'e2')
        self.assertEqual(got.rows[0]['entity_spans'][0]['token_start'],1)

    def test_unknown_labels_duplicate_alignment_and_missing_document_fail(self):
        docs=load_source([source_row(entities=[('e1',0,5,'unknown')],relations=[])])
        with self.assertRaisesRegex(ValueError,'inventory'):
            self.build(docs=docs)
        docs=load_source([source_row(entities=[('e1',0,5,'malware'),('e2',0,5,'tool')],relations=[])])
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            self.build(docs=docs)
        with self.assertRaisesRegex(ValueError,'identity'):
            self.build(windows=(ed.LogicalWindow(0,'other',0,0,3),))


class ExplicitSourceExclusionTests(unittest.TestCase):
    def load(self, rows, policy):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'source.json'; path.write_text(json.dumps(rows))
            return ed.load_label_studio_documents(path,excluded_unlabeled_regions=policy)

    def test_only_declared_unlabeled_unreferenced_region_is_excluded_with_coordinates(self):
        row=source_row(relations=[]); del row['annotations'][0]['result'][1]['value']['labels']
        docs=self.load([row],frozenset({('420','e2')}))
        self.assertEqual([e.entity_id for e in docs[0].entities],['e1'])
        excluded=docs[0].unlabeled_regions[0]
        self.assertEqual((excluded.entity_id,excluded.char_start,excluded.char_end),('e2',11,15))
        built=ed.build_encoder_windows(documents=docs,logical_windows=(ed.LogicalWindow(0,'420',0,0,3),),
            tokenizer=WhitespaceTokenizer(),inventory=inventory(),encoder_id='unit/encoder',
            encoder_revision='a'*40,max_length=8)
        self.assertEqual(built.audit['excluded_unlabeled_region_count'],1)
        self.assertTrue(any(e['reason']=='unlabeled_unreferenced_region' for e in built.exclusions))

    def test_referenced_labeled_or_nonexistent_exclusion_is_rejected(self):
        row=source_row(); del row['annotations'][0]['result'][1]['value']['labels']
        for rows,policy,error in [([row],frozenset({('420','e2')}),'endpoint'),
            ([source_row()],frozenset({('420','e2')}),'labeled'),
            ([source_row()],frozenset({('420','absent')}),'unused')]:
            with self.subTest(error=error),self.assertRaisesRegex(ValueError,error):
                self.load(rows,policy)
