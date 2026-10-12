"""Evidence and scope tests for the tool-free two-stage memory pipeline."""
import json
from types import SimpleNamespace
import unittest

from agent_core.memory.learning import (
    LearningEvidence, consolidation_messages, extraction_messages,
    extraction_from_dict, extraction_to_dict, parse_extraction, parse_proposals, transmitted_evidence, parse_consolidation,
)


def source(source_id='source', task_id='task', role='user', content='先核对来源再给结论'):
    return LearningEvidence(source_id, task_id, 'run', role, content, 'hash', 'succeeded')


def extracted(evidence, *, category='procedure'):
    return parse_extraction(json.dumps({'summary': '用户要求核对来源。', 'claims': [{
        'content': '核对来源再给结论', 'category': category, 'applicability': '当前项目的数据解释',
        'evidence_quotes': {item.source_id: item.content for item in evidence}}]}), evidence)


def proposals(source_ids=('source',), category='procedure'):
    return json.dumps({'proposals': [{'content': '核对来源再给结论', 'title': '来源核验',
        'category': category, 'applicability': '当前项目的数据解释', 'source_ids': list(source_ids)}]})


class LearningEvidenceTest(unittest.TestCase):
    def test_single_task_preference_is_retained_as_signal_but_not_published(self):
        result = extracted((source(),), category='preference')
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(parse_proposals(proposals(category='preference'), (result,)), ())
        second = extracted((source('second', 'task2'),), category='preference')
        learned = parse_proposals(proposals(('source', 'second'), 'preference'), (result, second))
        self.assertEqual(learned[0].evidence_task_ids, ('task', 'task2'))

    def test_two_runs_in_one_task_are_not_repeated_preference_evidence(self):
        a = extracted((source(),), category='preference')
        b = extracted((source('second'),), category='preference')
        self.assertEqual(parse_proposals(proposals(('source', 'second'), 'preference'), (a, b)), ())

    def test_invented_quote_and_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact supplied quotation'):
            parse_extraction(json.dumps({'summary': '', 'claims': [{'content': 'x', 'category': 'procedure',
                'applicability': 'project', 'evidence_quotes': {'source': '没有说过'}}]}), (source(),))
        with self.assertRaisesRegex(ValueError, 'unavailable evidence'):
            parse_proposals(proposals(('other-user-source',)), (extracted((source(),)),))

    def test_assistant_claim_alone_is_not_verification(self):
        self.assertEqual(extracted((source(role='assistant'),)).claims, ())

    def test_noop_schema_roundtrip_and_exact_duplicate(self):
        result = extracted((source(),))
        self.assertEqual(extraction_from_dict(extraction_to_dict(result)), result)
        self.assertEqual(parse_extraction('{"summary":"","claims":[]}', ()).claims, ())
        self.assertEqual(parse_proposals(proposals(), (result,),
            (SimpleNamespace(content='核对来源再给结论'),)), ())
        with self.assertRaises(ValueError):
            parse_proposals('{"proposals":[],"tool_calls":[]}', (result,))

    def test_input_budget_prioritizes_whole_human_evidence_and_redacts(self):
        inputs = (source('tool', role='tool', content='x' * 5000),
                  source(content='API_KEY=secret123 然后核查来源'))
        messages = extraction_messages(inputs, max_characters=2500)
        payload = json.loads(messages[1]['content'])
        self.assertEqual(len(payload['evidence']), 1)
        self.assertEqual(payload['evidence'][0]['role'], 'user')
        self.assertNotIn('secret123', messages[1]['content'])
        self.assertIn('untrusted', messages[0]['content'])

    def test_consolidation_receives_source_task_and_outcome(self):
        result = extracted((source(),))
        messages = consolidation_messages((result,), (), max_characters=10_000)
        payload = json.loads(messages[1]['content'])
        self.assertEqual(payload['source_context']['source']['task_id'], 'task')
        self.assertEqual(payload['source_context']['source']['outcome'], 'succeeded')
        with self.assertRaises(ValueError):
            consolidation_messages((result,), (), max_characters=10)

    def test_trimmed_source_cannot_be_cited_even_if_it_existed_in_archive(self):
        evidence = (source(), source('trimmed', role='tool', content='x' * 20_000))
        messages = extraction_messages(evidence, max_characters=3000)
        sent = transmitted_evidence(messages)
        self.assertEqual([item.source_id for item in sent], ['source'])
        with self.assertRaisesRegex(ValueError, 'exact supplied quotation'):
            parse_extraction(json.dumps({'summary': '', 'claims': [{'content': 'x', 'category': 'procedure',
                'applicability': 'project', 'evidence_quotes': {'trimmed': 'x'}}]}), sent)

    def test_generated_summary_cannot_promote_learned_record_into_confirmed_layer(self):
        record = SimpleNamespace(record_id='learned', version=1, status='learned', target='project', content='事实', source_ids=('source',))
        body = {'proposals': [], 'confirmed_summary': {'content': '已确认事实', 'source_refs': ['memory:learned:v1']},
                'learned_summary': {'content': '', 'source_refs': []}, 'user_confirmed_summary': {'content': '', 'source_refs': []}}
        with self.assertRaisesRegex(ValueError, 'crosses layer'):
            parse_consolidation(json.dumps(body), (), (record,))

    def test_user_summary_cannot_derive_from_project_confirmed_record(self):
        record = SimpleNamespace(record_id='project', version=1, status='confirmed', target='project',
                                 content='项目条件', source_ids=('source',))
        body = {'proposals': [], 'confirmed_summary': {'content': '', 'source_refs': []},
                'learned_summary': {'content': '', 'source_refs': []},
                'user_confirmed_summary': {'content': '全局条件', 'source_refs': ['memory:project:v1']}}
        with self.assertRaisesRegex(ValueError, 'crosses layer'):
            parse_consolidation(json.dumps(body), (), (record,))


if __name__ == '__main__':
    unittest.main()
