import socket
import sys

if __name__ == '__main__':

    # 假設錄好的影片存在這個路徑
    parm = "tests/result"  # 這是request需要自帶的參數

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9000))

    print(s.recv(1024).decode('utf-8'))  # 'Connected'訊息

    s.send(parm.encode('utf-8'))

    while True:
        message = s.recv(1024)
        if not message or message.decode('utf-8') == 'end':
            break
        print(message.decode('utf-8'))

    s.send(b'exit')

    s.close()
