import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

# Assuming you've already set your API keys as per the earlier section
os.environ["OPENAI_API_KEY"] = "..."

# Create a ChatOpenAI model instance
chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo")

# Chain One - Initial Email Analysis
initial_analysis_prompt = PromptTemplate(
    input_variables=["email_content", "subject", "red_flag", "attachments", "networkSenderIdentifier"],
    template="""
    Email Analysis Task:
    We are trying to classify emails as either legitimate or suspicious based on various factors like content, subject, red flags, attachments, and sender network identifiers.

    Please analyze the following email details:
    Subject: {subject}
    Content: {email_content}
    Attachments: {attachments}
    Sender: {networkSenderIdentifier}
    Red Flag: {red_flag}

    Provide an initial assessment based on the information provided:
    """
)
chain_one = LLMChain(llm=chatopenai, prompt=initial_analysis_prompt)

# Chain Two - Final Decision
final_decision_prompt = PromptTemplate(
    input_variables=["initial_analysis"],
    template="""
    Based on the initial analysis: {initial_analysis}
    Considering common patterns and signs, decide:
    Is this email legitimate or suspicious?
    """
)
chain_two = LLMChain(llm=chatopenai, prompt=final_decision_prompt)

# Sequential Chain Execution
sequential_email_chain = SequentialChain(
    input_variables=["email_content", "subject", "red_flag", "attachments", "networkSenderIdentifier"],
    chains=[chain_one, chain_two],
    verbose=True
)

# Example execution
result = sequential_email_chain.run(
    email_content="Congratulations! You've won a million dollars! Click here...",
    subject="You're a winner!",
    red_flag="No previous contact with sender",
    attachments="None",
    networkSenderIdentifier="unknown1234@example.com"
)

print(result)
