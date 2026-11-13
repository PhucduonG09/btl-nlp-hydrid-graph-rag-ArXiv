"""Shared retrieval interface; chunk IDs are paper IDs in abstract-only runs."""
from abc import ABC,abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    source: str

class Retriever(ABC):
    name='retriever'
    @abstractmethod
    def retrieve(self,query:str,k:int)->list[Hit]:...
