import sys
import json

def main():
    # Получаем аргументы командной строки
    args = sys.argv[1:]
    
    # Получаем параметры из stdin (если передаются)
    input_data = {}
    try:
        input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    except:
        pass
    
    # Собираем все параметры в один словарь
    params = {
        "command_line_args": args,
        "input_parameters": input_data,
        "received_parameters": {
            "args_count": len(args),
            "args_list": args,
            "stdin_data": input_data
        }
    }
    
    return {
        "message": "Parameters received successfully", 
        "status": "success",
        "parameters": params
    }

if __name__ == "__main__":
    result = main()
    print(f"RESULT: {result}")