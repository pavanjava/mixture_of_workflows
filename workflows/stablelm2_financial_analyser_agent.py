from llama_index.llms.ollama import Ollama
from llama_index.core import Document, SummaryIndex
from llama_index.core.workflow import (
    Context,
    StopEvent,
    step
)
from workflows.utils.workflow_events import TextExtractEvent
from workflows.base_financial_analyser_agent import BaseFinancialAnalyserAgent
import logging

logging.basicConfig(level=logging.INFO)


class Stablelm2FinancialAnalyserAgent(BaseFinancialAnalyserAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_llm = Ollama(model='stablelm2:latest',
                                base_url='http://localhost:11434',
                                temperature=0.2, request_timeout=300,
                                system_prompt="You are an AI Assistant to answer any questions from user. "
                                              "Strictly use the context provided to answer the user to answer the query"
                                              "don't consider your prior knowledge to answer, if you dont find the answer "
                                              "please respond 'I don't know.'")

    @step(pass_context=True)
    async def query_result(self, ctx: Context, ev: TextExtractEvent) -> StopEvent:
        """Get result with relevant text."""
        try:
            relevant_text = ev.relevant_text
            user_query = ctx.data["user_query"]

            documents = [Document(text=relevant_text)]
            # print(f"Documents: {documents}")
            index = SummaryIndex.from_documents(documents)
            query_engine = index.as_query_engine(llm=self.query_llm)
            result = query_engine.query(user_query)
            # print(f"Result: {result.response}")
            return StopEvent(result=str(result.response))
        except Exception as e:
            logging.error(str(e))
