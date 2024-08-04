# Local RAG Application

This repository contains code and resources to test and verify the examples provided in the article [To be replaced when published...].


## Prerequisites

* Python 3.9+ (It might work with previous but was tested with this one)
* Download and install [Ollama](https://ollama.com/download)
* Install the package listed in the [requirements.txt](./requirements.txt)

You will also need to download some models so they can run locally:

The LLM:

```
ollama run llama3.1
```

The text embeddings model:
```
ollama pull nomic-embed-text
```
