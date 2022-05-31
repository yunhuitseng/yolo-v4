import os
import socket
import threading
import time

import main as decode

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 如果 Server 伺服器端不正常關閉後再次啟動時可能會遇到error: [Errno 98] Address already in use
# 解決問題：
s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) # 啟用 keepalive
# s.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30000, 3000))

s.bind(("127.0.0.1", 9000)) # 我是用localhost:9000，到時候看實際上是要連去哪裡
s.listen(5)  # max 5 connections


def tcp(sock, addr):
    print("Accept new connection from %s:%s..." % addr)
    sock.send(b'Connected')
    while True:
        data = sock.recv(1024)  # receive data from client
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break

        # client 發送的請求要包含一個資料夾路徑
        filepath = data.decode('utf-8')
        
        for f in os.listdir(filepath):
            print(os.path.join(filepath, f))
            if f.endswith(".jpg"):
                decode.main(filepath)
        break

    sock.close()
    print("server closed")

  
if __name__ == '__main__':
    while True:
        # 接受一個新的連線
        data, addr = s.accept()
        print("server accepted")
        
        # 建立新的執行緒來處理這個連線
        t = threading.Thread(target=tcp, args=(data, addr))

        t.start()
