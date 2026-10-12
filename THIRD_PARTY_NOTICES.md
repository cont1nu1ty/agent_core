# Third-party dependencies

This project integrates installed dependencies through imports, subclassing and adapters. ReAct routing, filesystem operations and the upstream middleware are credited to their original projects.

| Dependency | Project | License in dependency metadata |
| --- | --- | --- |
| LangChain, langchain-core, langchain-openai | [LangChain](https://github.com/langchain-ai/langchain) | MIT |
| LangGraph, langgraph-checkpoint | [LangGraph](https://github.com/langchain-ai/langgraph) | MIT |
| Deep Agents | [Deep Agents](https://github.com/langchain-ai/deepagents) | MIT |
| Pydantic | [Pydantic](https://github.com/pydantic/pydantic) | MIT |
| HTTPX | [HTTPX](https://github.com/encode/httpx) | BSD-3-Clause |
| typing-extensions | [typing-extensions](https://github.com/python/typing_extensions) | PSF-2.0 |

Versions are declared in [pyproject.toml](pyproject.toml) and resolved in [uv.lock](uv.lock). The installed distributions carry their own license texts and notices, including transitive dependencies. This table is an attribution guide, not a replacement for those texts.

No project license has been selected for the extracted Agent Core source. This file does not grant MIT or other rights for that source.
