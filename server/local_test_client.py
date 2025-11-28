import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    server_url = os.getenv("SERVER_URL", "http://localhost:8001")
    thread_id = str(uuid.uuid4())
    
    print("=" * 50)
    print("🌍 Lifestyle Travel Assistant - Консольный клиент")
    print("=" * 50)
    print(f"Сервер: {server_url}")
    print(f"Thread ID: {thread_id}")
    print("-" * 50)
    print("Команды:")
    print("  /clear - очистить историю диалога")
    print("  /quit или /exit - выйти")
    print("=" * 50)
    print()
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            # Обработка команд
            if user_input.lower() in ["/quit", "/exit"]:
                print("👋 До свидания!")
                break
            
            if user_input.lower() == "/clear":
                thread_id = str(uuid.uuid4())
                print(f"✅ История очищена. Новый Thread ID: {thread_id}")
                print()
                continue
            
            # Отправка запроса на сервер
            try:
                response = requests.post(
                    f"{server_url}/chat",
                    json={
                        "message": user_input,
                        "thread_id": thread_id
                    },
                    timeout=120.0  # Увеличенный таймаут для сложных запросов
                )
                
                if response.status_code == 200:
                    data = response.json()["response"]
                    print(f"\n🤖 Ассистент: {data}\n")
                else:
                    print(f"\n❌ Ошибка сервера: {response.status_code}")
                    print(f"Ответ: {response.text}\n")
                    
            except requests.exceptions.Timeout:
                print("\n⏱️ Превышено время ожидания ответа от сервера\n")
            except requests.exceptions.ConnectionError:
                print(f"\n🔌 Нет связи с сервером ({server_url}). Проверьте, запущен ли сервер.\n")
            except Exception as e:
                print(f"\n❌ Произошла ошибка: {str(e)}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except EOFError:
            print("\n👋 До свидания!")
            break


if __name__ == "__main__":
    main()
