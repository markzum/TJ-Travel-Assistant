from langchain_openai import ChatOpenAI


def create_llm():
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="fake-key",
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.7
    )
    return llm

req = "а что ты думаешь?"
llm = create_llm()
print(f'Запрос: {req}\nОтвет:')
print(llm.invoke(req).content)
