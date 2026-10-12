"""Public streaming API exercised with an in-process HTTP transport only."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import InternalServerError

from agent_core.models.requests import bounded_call_model, output_limit_kwargs


class ModelRequestLimitsTest(unittest.TestCase):
    def setUp(self):
        # SDK request headers probe uname through platform.platform. Keep that
        # unrelated OS metadata deterministic; the audit guard still forbids all
        # subprocesses/network while the real SDK uses our explicit transport.
        platform_metadata = patch('platform.platform', return_value='Linux-test')
        platform_metadata.start()
        self.addCleanup(platform_metadata.stop)

    def test_explicit_gateway_extra_body_cannot_override_summary_output_limit(self):
        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            chunk = {'id': 'fake-id', 'object': 'chat.completion.chunk', 'created': 0,
                     'model': 'fake-model', 'choices': [{'index': 0, 'delta': {'content': '摘要'},
                                                       'finish_reason': 'stop'}]}
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content=('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())
        async def exercise():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                model = ChatOpenAI(model='fake-model', api_key='unit-test-not-a-secret',
                    base_url='https://fake.invalid/v1', http_async_client=client, max_retries=0,
                    extra_body={'max_tokens': 8192, 'thinking': {'type': 'disabled'}})
                wrapped = SimpleNamespace(chat_model=model)
                chunks = [chunk async for chunk in model.astream([HumanMessage('摘要请求')],
                          **output_limit_kwargs(wrapped, 1536))]
                self.assertTrue(chunks)
                self.assertEqual(model.extra_body['max_tokens'], 8192)
        asyncio.run(exercise())
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['max_tokens'], 1536)
        self.assertEqual(requests[0].get('max_completion_tokens', 1536), 1536)
        self.assertEqual(requests[0]['thinking'], {'type': 'disabled'})

    def test_generic_models_and_both_explicit_limit_names(self):
        self.assertEqual(output_limit_kwargs(object(), 2500), {'max_tokens': 2500})
        extra = {'max_tokens': 7000, 'max_completion_tokens': 8000, 'custom': True}
        result = output_limit_kwargs(SimpleNamespace(extra_body=extra), 2500)
        self.assertEqual(result['extra_body'], {'max_tokens': 2500, 'max_completion_tokens': 2500, 'custom': True})
        self.assertEqual(extra['max_tokens'], 7000)

    def test_auxiliary_view_disables_sdk_retries_without_mutating_foreground_client(self):
        requests = []
        async def handle(request):
            requests.append(request)
            return httpx.Response(500, json={'error': {'message': 'fake temporary failure', 'type': 'server_error'}})
        async def exercise():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                model = ChatOpenAI(model='fake-model', api_key='unit-test-not-a-secret',
                    base_url='https://fake.invalid/v1', http_async_client=client, max_retries=2)
                view = bounded_call_model(model)
                with self.assertRaises(InternalServerError):
                    async for _ in view.astream([HumanMessage('摘要请求')]):
                        self.fail('500 response cannot yield a summary')
                self.assertEqual(model.root_async_client.max_retries, 2)
                self.assertEqual(view.root_async_client.max_retries, 0)
                self.assertFalse(client.is_closed)
        asyncio.run(exercise())
        self.assertEqual(len(requests), 1)


if __name__ == '__main__':
    unittest.main()
