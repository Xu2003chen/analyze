# 服务端
import http.server
import socketserver
import json
import threading
from datetime import datetime
import os
import atexit
import signal
import socket


# 设置端口
PORT = 1111
current_dir = os.getcwd()
# 聊天记录文件路径
MESSAGE_FILE = os.path.join(current_dir, "data", "others", "message.json")

# 全局消息列表
messages = []
# 锁，防止并发读写冲突
msg_lock = threading.Lock()


def load_messages():
    # 加载历史消息
    global messages
    with msg_lock:
        if os.path.exists(MESSAGE_FILE):
            with open(MESSAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    messages = data
                    print(f"已加载 {len(data)} 条历史消息")
                else:
                    messages = []


def save_messages():
    """关闭时保存消息到 message.json"""
    with msg_lock:
        try:
            with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"\n聊天记录已保存到: {MESSAGE_FILE}")
        except IOError as e:
            print(f"保存聊天记录: {e}")


# ==================== 服务器逻辑 ====================


class ChatRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/post_msg":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                username = data.get("username", "匿名")
                text = data.get("text", "").strip()

                if not text:
                    self.send_response(400)
                    self.end_headers()
                    return

                new_msg = {
                    "username": username,
                    "text": text,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }

                with msg_lock:
                    messages.append(new_msg)
                    # 保留最近500条
                    if len(messages) > 500:
                        messages.pop(0)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
        else:
            self.send_response(404)
            self.end_headers()


    def do_GET(self):
        if self.path == "/get_msgs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with msg_lock:
                self.wfile.write(json.dumps(messages).encode("utf-8"))
        else:
            super().do_GET()


# ==================== 开服 ====================

if __name__ == "__main__":
    # 1. 启动时加载历史消息
    load_messages()

    # 2. 注册退出函数：程序结束时自动保存
    atexit.register(save_messages)

    # 3. 处理 Ctrl+C 和其他终止信号
    def signal_handler(signum, frame):
        save_messages()
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    # 4. 启动服务器
    with socketserver.TCPServer(("", PORT), ChatRequestHandler) as httpd:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(f"聊天服务器启动在端口 {PORT}")
        print(f"访问地址: http://{ip}:{PORT}")
        print(f"消息接口: http://{ip}:{PORT}/get_msgs")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n停止...")
