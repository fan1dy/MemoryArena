import os
from typing import Dict, Callable, NamedTuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from memory_systems import (
    LongContextMemorySystem,
    MirixMemorySystem,
    Mem0MemorySystem,
    LettaMemorySystem,
    RAGMemorySystem,
    MemoRAGMemorySystem,
    GraphRAGMemorySystem,
    AMemMemorySystem,
    LightMemMemorySystem,
    ReasoningBankMemorySystem,
    ZepMemorySystem,
    Mem0LocalMemorySystem,
    LettaLocalMemorySystem,
)


load_dotenv()

# ---------- Request schemas ----------
class InitializeRequest(BaseModel):
    user_id: str
    memory_system_name: str


class AddRequest(BaseModel):
    user_id: str
    chunk: str
    memory_system_name: str


class QueryRequest(BaseModel):
    user_id: str
    question: str
    memory_system_name: str

class ActRequest(BaseModel):
    user_id: str
    prompt: str
    memory_system_name: str

# ---------- Memory/Agent implementations (unified) ----------
_ALL_FACTORIES = {
    "mirix": MirixMemorySystem,
    "long_context": LongContextMemorySystem,
    "mem0": Mem0MemorySystem,
    "mem0-g": lambda: Mem0MemorySystem(enable_graph=True) if Mem0MemorySystem else None,
    "mem0-local": Mem0LocalMemorySystem,
    "mem0-local-g": lambda: Mem0LocalMemorySystem(enable_graph=True) if Mem0LocalMemorySystem else None,
    "letta": LettaMemorySystem,
    "letta-local": LettaLocalMemorySystem,
    "rag": RAGMemorySystem,
    "memorag": MemoRAGMemorySystem,
    "graphrag": GraphRAGMemorySystem,
    "amem": AMemMemorySystem,
    "lightmem": LightMemMemorySystem,
    "reasoningbank": ReasoningBankMemorySystem,
    "zep": ZepMemorySystem,
}
MEMORY_FACTORIES: Dict[str, Callable[[], object]] = {
    k: v for k, v in _ALL_FACTORIES.items() if v is not None
}

# Memory systems that hold a process-wide lock (e.g. local Qdrant) must be
# instantiated only once and shared across all user_ids.
SINGLETON_MEMORY_SYSTEMS = {"mem0-local", "mem0-local-g"}
SHARED_INSTANCES: Dict[str, object] = {}


# ---------- FastAPI wiring ----------
app = FastAPI(title="Memory Agent Server")


class MemorySystemEntry(NamedTuple):
    name: str
    system: object


MEMORY_SYSTEMS: Dict[str, MemorySystemEntry] = {}


def _get_memory(user_id: str, memory_system: str):
    entry = MEMORY_SYSTEMS.get(user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="User not initialized")
    if entry.name != memory_system:
        raise HTTPException(status_code=400, detail="Mismatched memory_system for user")
    return entry.system


@app.post("/memory/initialize")
def initialize(req: InitializeRequest):
    name = req.memory_system_name
    if name in {"bm25", "text-embedding-3-small"}:
        memory_system = RAGMemorySystem(retrieval_method=name)
    # elif name == "graphrag":
    #     if not os.getenv("GRAPHRAG_LOCAL_DIR"):
    #         raise HTTPException(status_code=400, detail="GRAPHRAG_LOCAL_DIR is required to initialize GraphRAG.")
    #     memory_system = GraphRAGMemorySystem(local_dir=os.getenv("GRAPHRAG_LOCAL_DIR"))
    elif name in {"reasoningbank"}:
        memory_system = ReasoningBankMemorySystem(user_id=req.user_id)
    else:
        factory = MEMORY_FACTORIES.get(name)
        if factory is None:
            raise HTTPException(status_code=400, detail=f"Unsupported memory_system: {name}")
        if name in SINGLETON_MEMORY_SYSTEMS:
            if name not in SHARED_INSTANCES:
                SHARED_INSTANCES[name] = factory()
            memory_system = SHARED_INSTANCES[name]
        else:
            memory_system = factory()
    MEMORY_SYSTEMS[req.user_id] = MemorySystemEntry(name=name, system=memory_system)
    return {"status": "ok", "user_id": req.user_id, "memory_system_name": name}


@app.post("/memory/add")
def add(req: AddRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    response = memory_system.add_chunk(req.chunk, user_id=req.user_id)
    outputs = {"status": "ok", "user_id": req.user_id}
    outputs['response'] = response
    return outputs


@app.post("/memory/wrap_user_prompt")
def wrap_user_prompt(req: QueryRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    prompt = memory_system.wrap_user_prompt(req.question, user_id=req.user_id)
    return {"status": "ok", "user_id": req.user_id, "prompt": prompt}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
