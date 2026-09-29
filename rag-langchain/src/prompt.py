from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant.

Answer the user's question based only on the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


context = """
Subject: Product setup
Description: My product is making strange noises and not functioning properly.
I suspect there might be a hardware issue.
Status: Open
Priority: Medium
Channel: Chat
"""

question = "I have an issue with product setup. What should I do?"


final_prompt = prompt.format(
    context=context,
    question=question
)


print(final_prompt)