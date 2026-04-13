# CopilotJ: A Conversational Multi-agent System for Intelligent and Efficient Bioimage Analysis
[![Project Page](https://img.shields.io/badge/Project%20Page-copilotj.chat-blue?style=flat-square)](https://copilotj.chat/)
[![User Manual](https://img.shields.io/badge/User%20Manual-docs-green?style=flat-square)](https://copilotj.chat/#/manual)
[![Demos](https://img.shields.io/badge/Demos-YouTube-red?style=flat-square&logo=youtube)](https://www.youtube.com/playlist?list=PLXIvvwpNOx3cg0OSE6mh-BxzaO0pMEVpb)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)](LICENSE)

Learn more on the [Project Page](https://copilotj.chat/), follow the [User Manual](https://copilotj.chat/#/manual) to get started, or watch the [Video Demos](https://www.youtube.com/playlist?list=PLXIvvwpNOx3cg0OSE6mh-BxzaO0pMEVpb) to see CopilotJ in action.


## About

**CopilotJ** turns natural-language requests into executable, verifiable bioimage analysis workflows by coordinating Fiji/ImageJ ecosystem, Python scientific libraries, and deep-learning models.

CopilotJ comprises three major components:

- **Web Frontend** — A browser-based chat interface through which users interact with CopilotJ. It sends requests to the multi-agent backend and receives streaming responses via RESTful APIs.
- **Multi-agent Backend** — Orchestrates agents and tools, integrates Python and deep-learning environments with a local database, connects to multiple LLM providers, and retrieves community knowledge sources.
- **CopilotJ Bridge** — A Fiji plugin that communicates with the backend via a bidirectional WebSocket API, enabling real-time exchange of system status and commands.

**Visit [copilotj.chat](https://copilotj.chat/#/) for more information and demos.**

## Installation & Configuration

For full installation and configuration instructions, including environment setup, API key configuration, and how to run each component, please refer to the [User Manual](https://copilotj.chat/#/manual).


## License

This project is licensed under the [Apache License 2.0](LICENSE).
