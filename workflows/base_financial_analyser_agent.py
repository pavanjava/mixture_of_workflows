from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    step
)
from workflows.utils.workflow_events import RetrieveEvent, TextExtractEvent, RelevanceEvalEvent
from rag_core.rag_operations import RAGOperations
from workflows.utils.prompts import DEFAULT_RELEVANCY_PROMPT_TEMPLATE
import logging

logging.basicConfig(level=logging.INFO)


class BaseFinancialAnalyserAgent(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_llm = Ollama(model='llama3.1', base_url='http://localhost:11434', temperature=0.8,
                               request_timeout=300)
        Settings.embed_model = OllamaEmbedding(model_name='all-minilm:33m', base_url='http://localhost:11434')

    @step(pass_context=True)
    async def ingest_and_retrieve_docs(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        try:
            """Ingest step (for ingesting docs and initializing prompt templates)."""
            rag_ops = RAGOperations()
            ctx.data['rag_ops'] = rag_ops
            ctx.data["relevancy_pipeline"] = QueryPipeline(
                chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, self.eval_llm]
            )

            """Retrieve the relevant nodes from the user query."""
            rag_ops: RAGOperations = ctx.data['rag_ops']
            user_query = ev.get("user_query")
            retrieved_nodes = rag_ops.create_retriever().retrieve(str_or_query_bundle=user_query)
            ctx.data["retrieved_nodes"] = retrieved_nodes
            ctx.data["user_query"] = user_query
            return RetrieveEvent(retrieved_nodes=retrieved_nodes)
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def evaluate_relevance(self, ctx: Context, ev: RetrieveEvent) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved nodes with the user query."""
        try:
            retrieved_nodes = ev.retrieved_nodes
            user_query = ctx.data["user_query"]

            relevancy_results = []
            for node in retrieved_nodes:
                relevancy = ctx.data["relevancy_pipeline"].run(context_str=node.text, query_str=user_query)
                relevancy_results.append(relevancy.message.content.lower().strip())

            ctx.data["relevancy_results"] = relevancy_results
            return RelevanceEvalEvent(relevant_results=relevancy_results)
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def extract_relevant_text(self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        try:
            retrieved_nodes = ctx.data["retrieved_nodes"]
            relevancy_results = ev.relevant_results

            relevant_texts = [
                retrieved_nodes[i].text
                for i, result in enumerate(relevancy_results)
                if "**yes**" in result
            ]

            result = "\n".join(relevant_texts)
            return TextExtractEvent(relevant_text=result)
        except Exception as e:
            logging.error(str(e))
