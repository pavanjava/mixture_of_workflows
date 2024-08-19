from workflows.phi3_financial_analyser_agent import Phi3FinancialAnalyserAgent
from workflows.gemma2_financial_analyser_agent import Gemma2FinancialAnalyserAgent
from workflows.qwen2_financial_analyser_agent import Qwen2FinancialAnalyserAgent
from workflows.stablelm2_financial_analyser_agent import Stablelm2FinancialAnalyserAgent
from workflows.financial_aggregator_agent import FinancialAggregatorAgent
import nest_asyncio

# Apply the nest_asyncio
nest_asyncio.apply()


async def main():
    w1 = Phi3FinancialAnalyserAgent(timeout=300, verbose=True)
    w2 = Gemma2FinancialAnalyserAgent(timeout=300, verbose=True)
    w3 = Qwen2FinancialAnalyserAgent(timeout=300, verbose=True)
    w4 = Stablelm2FinancialAnalyserAgent(timeout=300, verbose=True)
    w5 = FinancialAggregatorAgent(timeout=300, verbose=True)
    user_query = "what are Fourth Quarter Highlights?"

    result_1 = await w1.run(user_query=user_query)
    result_2 = await w2.run(user_query=user_query)
    result_3 = await w3.run(user_query=user_query)
    result_4 = await w4.run(user_query=user_query)
    print(f"Phi3 Result: {result_1}")
    print(f"Gemma2 Result: {result_2}")
    print(f"Qwen2 Result: {result_3}")
    print(f"StableLM2 Result: {result_4}")

    summary = await w5.run(slm_results=[result_1, result_2, result_3, result_4])

    print(f"Final Summary: {summary}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(main=main())
