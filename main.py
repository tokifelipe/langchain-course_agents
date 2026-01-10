from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(r"C:\Users\FelipeMuñozVargas\langchain-course_agents\.env")


from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
#Para que se parsee la respuesta del agente solo al final de las iteraciones y no en cada paso, se crea un LLM estructurado con la clase AgentResponse  
#Usa la llamada a la función cuando está disponible , falling back to parsing when not
structuctured_llm=llm.with_structured_output(AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
#Crea el ejecutor del agente: Toma el agente React y le da acceso a las herramientas definidas, verbose : True para ver el proceso detalladamente
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#Crea un estractor que toma el resultado completo del agente y solo saca la parte de "output" 
extract_output = RunnableLambda(lambda x: x["output"])
#Parser: Convierte el texto de salida del agente en un objeto Pydantic definido en schemas.py llamado AgentResponse s
#Primero se ejecuta el agente, luego se extrae la salida y finalmente se parsea a un objeto Pydantic
chain = agent_executor | extract_output | structuctured_llm


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()