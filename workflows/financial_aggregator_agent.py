from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent
from llama_index.core import SummaryIndex, Document
from llama_index.llms.ollama import Ollama
import logging

logging.basicConfig(level=logging.INFO)


class FinancialAggregatorAgent(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_llm = Ollama(model='llama3.1',
                                base_url='http://localhost:11434',
                                temperature=0.7, request_timeout=300,
                                system_prompt="You are an AI Assistant to summarize any data provided. "
                                              "Strictly use the context provided to summarize don't consider your "
                                              "prior knowledge for summarization, if you dont find the answer "
                                              "please respond 'I don't know.'")

    @step(pass_context=False)
    async def extract_relevant_text(self, ev: StartEvent) -> StopEvent:
        """Get result with relevant text."""
        try:
            user_query = "fetch summary of the financial data in bullet points"
            slm_results = ev.slm_results
            documents = [Document(text=''.join(slm_results))]
            index = SummaryIndex.from_documents(documents)
            query_engine = index.as_query_engine(llm=self.query_llm)
            result = query_engine.query(user_query)
            # print(f"Result: {result.response}")
            return StopEvent(result=str(result.response))
        except Exception as e:
            logging.error(str(e))
