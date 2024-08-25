# Langchain
- In this first we can learn `langchain` and then we can build a QnA chat_bot.

# Chat Models
- List of All the chat models
- ![ChatModels](https://python.langchain.com/v0.2/docs/integrations/chat/)

- Here we can use googele generative ai.
- python```
    from langchain_google_genai import GoogleGenerativeAI
    from dotenv import load_dotenv
    import os


    load_dotenv()
    api=os.environ["google_api"]

    print("Loading Model...")
    model= GoogleGenerativeAI(
        google_api_key=api,
        model="gemini-1.5-pro",
        temperature=0.5,
        max_output_tokens=20
    )

    print("Getting Response...")
    response=model.invoke("Which city is the biggest city in pakistan")

    print("Response:")
    print(response)
    ```
#  Basic Conversation Bot
- python ```
    from langchain_google_genai import GoogleGenerativeAI
    from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
    import os 
    from dotenv import load_dotenv


    load_dotenv()
    api=os.environ["google_api"]

    print('Loading Model...')
    model=GoogleGenerativeAI(google_api_key=api,model="gemini-1.5-pro",temperature=0.5,max_output_tokens=50)


    print("Setting Mesaages..")
    messages=[
        SystemMessage(content="You are a helpful chatbot"),
        HumanMessage(content="can you write some python program for me program is to add 2 nbr program should be dynamic"),
    ]


    print("Generating Response...")
    response=model.invoke(messages)

    print("Ai Response: ",response)

    print("Setting AI messages...")
    messages=[
        SystemMessage(content="you are a assistant to solve basic math problems"),
        HumanMessage(content="what is the product of 4 and 4"),
        AIMessage(content="product of 4 and 4 is equal to 16"),
        HumanMessage(content="what is the product of 7800 and 9000")
    ]

    print("Generating New Response...")
    response=model.invoke(messages)
    print("Ai Response: ",response)
    ```