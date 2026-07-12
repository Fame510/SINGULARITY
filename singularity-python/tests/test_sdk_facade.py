import pytest

import singularity


def test_version_exposed():
    assert isinstance(singularity.__version__, str)


def test_distributed_llm_requires_real_engine():
    llm = singularity.DistributedLLM(model="meta-llama/Meta-Llama-3-8B-Instruct",
                                     oracle_enabled=True)
    with pytest.raises(NotImplementedError):
        llm.generate("hello")


def test_distributed_llm_uses_injected_engine():
    class Echo:
        def generate(self, prompt, max_tokens=256, **kw):
            return f"ECHO:{prompt}"
    llm = singularity.DistributedLLM(model="m", engine=Echo())
    assert llm.generate("hi") == "ECHO:hi"
    assert llm.stats["total_tokens"] == 256
