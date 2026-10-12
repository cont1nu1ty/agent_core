"""Fill native async gaps in the pinned Deep Agents StoreBackend.

The upstream backend owns file formats and matching semantics. Its read/write/
edit/delete methods already await BaseStore; directory/search/download/upload
still inherit blocking fallbacks in 0.7.13. Keep those I/O calls on this loop.
"""

import base64

from deepagents.backends import StoreBackend
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse
from deepagents.backends.utils import create_file_data, file_data_to_string
from langgraph.store.memory import InMemoryStore


class AsyncStoreBackend(StoreBackend):
    async def _listing(self) -> StoreBackend:
        namespace = self._get_namespace()
        items = await self._asearch_store_paginated(self._get_store(), namespace)
        # An ephemeral projection for the upstream pure matching/formatting code;
        # all authoritative reads above use the host's asynchronous store.
        snapshot = InMemoryStore()
        for item in items:
            snapshot.put(namespace, item.key, item.value)
        return StoreBackend(store=snapshot, namespace=lambda _runtime: namespace)

    async def als(self, path):
        return (await self._listing()).ls(path)

    async def agrep(self, pattern, path=None, glob=None, *, max_count=None):
        return (await self._listing()).grep(pattern, path, glob, max_count=max_count)

    async def aglob(self, pattern, path=None):
        return (await self._listing()).glob(pattern, path)

    async def adownload_files(self, paths):
        store, namespace = self._get_store(), self._get_namespace()
        responses = []
        for path in paths:
            item = await store.aget(namespace, path)
            if item is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue
            file_data = self._convert_store_item_to_file_data(item)
            content = file_data_to_string(file_data)
            data = (base64.standard_b64decode(content) if file_data["encoding"] == "base64"
                    else content.encode("utf-8"))
            responses.append(FileDownloadResponse(path=path, content=data, error=None))
        return responses

    async def aupload_files(self, files):
        store, namespace = self._get_store(), self._get_namespace()
        responses = []
        for path, content in files:
            try:
                text, encoding = content.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                text, encoding = base64.standard_b64encode(content).decode("ascii"), "base64"
            value = self._convert_file_data_to_store_value(create_file_data(text, encoding=encoding))
            await store.aput(namespace, path, value)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses
