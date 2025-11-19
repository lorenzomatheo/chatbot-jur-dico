from __future__ import annotations
import os
import sys
from typing import List

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from deepagents import create_deep_agent
from langchain_core.tools import BaseTool

# ------------------------------------------------------------------
# 1. Import our CLT tool
# ------------------------------------------------------------------
from tools.legal_tools import clt_rag

SYSTEM_PROMPT = """
Você é um advogado trabalhista especializado na CLT.

Se a resposta não estiver presente no contexto, diga claramente:
"Não encontrei essa informação na CLT fornecida."

Instruções:
- Use **sempre** a ferramenta `clt_rag` quando a dúvida envolver CLT. 
- Traga artigos específicos, trechos relevantes e explique o raciocínio passo a passo.
- Não invente leis, artigos ou jurisprudências.
- Não traga conhecimento externo.
- Registre rascunhos em `./scratch/<uuid>.md` sempre que precisar planejar ou comparar hipóteses.
"""

# ------------------------------------------------------------------
# 2. Use of DeepAgents library
# ------------------------------------------------------------------
      

def build_legal_deep_agent(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0):
    model = ChatGroq(model=model_name, temperature=temperature,api_key=os.getenv("GROQ_API_KEY"))
    return create_deep_agent(
        tools=[clt_rag],
        system_prompt=SYSTEM_PROMPT,
        model=model,
        # subagents=[],        # future
        # middleware=[...],    # future
    )

# ------------------------------------------------------------------
# 4. CLI smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = build_legal_deep_agent()
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Qual o prazo para homologar a rescisão?"}
        ]
    })
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "content"):
        print(last_msg.content)
    else:
        print(last_msg.get("content"))