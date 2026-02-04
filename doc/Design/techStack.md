## Do not create excessive folders, keep the directory easy to understand.
1. LLM API, in src/llm provide interface for ollama, fireworks, openai, claude, and gemini api interfaces. and have a unified completion interface.
2. You can reference alphaevolve and PFSagent but come up with the best design based on core novelty, that it should agentic centric tuner instead of tuning workflow with embedded LLM calls. 
3. should provide interface for agent to call all tools, manage memory/knowledge pool, run microbenchmark, launch realworkload, collect metrics/logs, enforce SLA and compile configs etc.
4. Agent Itself should be modular, and should be able to interact and config NCCL you can reference autoCCL, agent should be able to be used as a nccl ext-net ext-tuner
5. realworkload is to train qwen30B or llama2-70B model on multiple nodes with different topology and scale.

Whole project is with NCCL don't worry about other CCL.
