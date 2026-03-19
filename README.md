Implementation of RAG System

The System consists of:
  - Word format Handler (doc, docx)
  - Document Indexer (Use txtai)
  - Entities extractor with graph saving (Use Gliner, Gliner2, Regex and nx.Graph)
  - Triplets extractor with another graph saving (Use nx.Graph and triplet extractor based on LLM)
  - Prompt Assembler include prompt constructor
  - Base LLM, which can be Local or Remote from public service (Ollama, GigaChat or remote service based on OpenAI client like OpenRouter)
Also add Metrics:
  - Simple metrics like bert-score, blue, precision, recall, meteor and other
  - LLM Judge metrics with prompt evaluate on correctness, faithfulness, completeness, relevance
    
All parameters are being configured with use config.yaml and interactive cli choice.
Interactive cli menu permit modify configuration and enter query to system.
All modules can be choose by desire and anyone might gather own RAG construction.
For Local LLM implemented GPU memory distribution with diffrent approch include accelerator.

System start with command: python main.py --run

Response and Metrics displayed in cli menu.

Database don't realize, RAG system work in memory.
