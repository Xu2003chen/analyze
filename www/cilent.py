# 客户端
import requests
import threading
import time
import getpass
import os
from queue import Queue, Empty

# 地址
SERVER_URL = "http://192.168.1.234:1111"
POLL_INTERVAL = 1.0

# 线程间通信队列
message_queue = Queue()  # 用于接收新消息
input_queue = Queue()  # 用于接收用户输入完成的消息
display_queue = Queue()  # 用于主线程发送重绘指令

running = True
messages = []
input_prompt_shown = False


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_interface(msgs, show_input=True):
    clear_screen()
    print("欢迎（输入 'quit' 退出，'clear' 清屏）\n")
    print("-" * 60)
    for msg in msgs:
        print(f"[{msg['timestamp']}] {msg['username']}: {msg['text']}")
    print("-" * 60)
    if show_input:
        print("发送: ", end="", flush=True)


def fetch_messages():
    try:
        res = requests.get(f"{SERVER_URL}/get_msgs", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"\n消息拉取失败: {e}", flush=True)
    return []


def poller_thread():
    """后台轮询消息线程"""
    global messages, running
    last_count = len(messages)
    while running:
        new_msgs = fetch_messages()
        if len(new_msgs) > last_count:
            for msg in new_msgs[last_count:]:
                message_queue.put(msg)
            last_count = len(new_msgs)
            messages = new_msgs  # 更新消息
            display_queue.put("refresh")  # 触发重绘
        time.sleep(POLL_INTERVAL)


def input_thread():
    global running
    while running:
        try:
            user_input = input()
            input_queue.put(user_input)
        except EOFError:
            break
        except Exception as e:
            if running:
                input_queue.put(f"__ERROR__: {str(e)}")
            break


def main():
    global running, messages

    clear_screen()
    username = input("请输入用户名: ").strip()
    if not username:
        username = getpass.getuser()
    print(f"欢迎 {username}！")

    # 初始化消息
    messages = fetch_messages()
    print_interface(messages)

    # 启动后台线程
    poller = threading.Thread(target=poller_thread, daemon=True)
    poller.start()

    inputer = threading.Thread(target=input_thread, daemon=True)
    inputer.start()

    # 主循环：处理消息和输入
    while running:
        try:
            # 1. 处理新消息（刷新界面）
            try:
                msg = message_queue.get_nowait()
                print(f"\n[{msg['timestamp']}] {msg['username']}: {msg['text']}")
            except Empty:
                pass

            try:
                cmd = display_queue.get_nowait()
                if cmd == "refresh":
                    print_interface(messages, show_input=True)
            except Empty:
                pass

            # 2. 处理用户输入
            try:
                user_input = input_queue.get(timeout=0.1)
                if user_input.startswith("__ERROR__"):
                    print(f"\n输入错误: {user_input[11:]}")
                    print_interface(messages)
                    continue

                text = user_input.strip()
                if text.lower() == "quit":
                    print("\n再见！")
                    running = False
                    break
                elif text.lower() == "clear":
                    print("\n已清屏")
                    # 不真清，只是重新显示
                    print_interface(messages)
                elif text:
                    try:
                        res = requests.post(
                            f"{SERVER_URL}/post_msg",
                            json={"username": username, "text": text},
                        )
                        if res.status_code != 200:
                            print(f"\n发送失败: {res.text}")
                    except Exception as e:
                        print(f"\n连接服务器失败: {e}")
                    print_interface(messages)  # 重新显示输入提示
                else:
                    print_interface(messages)  # 空输入，重新提示

            except Empty:
                continue  # 继续循环

        except KeyboardInterrupt:
            print("\n\n再见！")
            running = False
            break
        except Exception as e:
            running = False
            break


if __name__ == "__main__":
    main()
