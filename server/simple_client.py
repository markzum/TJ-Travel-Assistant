import requests
import sys
import uuid


def main():
    server_url = "http://localhost:8001"
    
    # Генерация уникального thread_id для этой сессии
    thread_id = str(uuid.uuid4())
    
    print("Lifestyle Travel Agent")
    print(f"Session ID: {thread_id}")
    print("Введите 'выход' для завершения\n")
    
    # Проверка доступности сервера
    try:
        response = requests.get(f"{server_url}/health", timeout=5.0)
        if response.status_code != 200:
            print("❌ Сервер недоступен. Убедитесь, что server.py запущен.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к серверу. Запустите server.py перед запуском клиента.")
        print("   Для запуска сервера: python server.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при проверке сервера: {e}")
        sys.exit(1)
    
    print("✅ Подключено к серверу\n")
    
    while True:
        user_input = input("Вы: ").strip()
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break
        
        if not user_input:
            continue
        
        try:
            # Отправка запроса на сервер с только message и thread_id
            response = requests.post(
                f"{server_url}/chat",
                json={
                    "message": user_input,
                    "thread_id": thread_id
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_response = data["response"]
                
                print(f"\nАссистент: {assistant_response}\n")
            else:
                print(f"❌ Ошибка сервера: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("❌ Превышено время ожидания ответа от сервера")
        except requests.exceptions.ConnectionError:
            print("❌ Потеряно соединение с сервером")
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
