"""
DeepAgent-compatible tools.
Each tool returns JSON-serialisable dict and gets an auto-generated schema.
"""
from langchain_core.tools import tool
from legal_rag import answer_about_clt
from typing import Dict

@tool
def clt_rag(question: str) -> Dict:
    """
    Responde dúvidas trabalhistas com base na CLT, retornando resposta e citações.
    """
    return answer_about_clt(question)