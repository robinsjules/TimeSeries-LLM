from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def generate_insights(forecast, metrics):
    llm = OpenAI(temperature=0.5, model_name="gpt-4")
    
    prompt_template = """
    As a financial analyst, explain this market forecast and strategy performance:
    
    Forecast Summary:
    {forecast}
    
    Performance Metrics:
    {metrics}
    
    Provide key insights and potential driving factors in 3 bullet points:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["forecast", "metrics"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(forecast=forecast, metrics=metrics)