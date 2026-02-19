import socket
import struct
import threading
import time
import random

class MatrixTcpServer:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clients = []
        self.running = False
        
        # 예제용 3x3 행렬들 (순수 Python 리스트)
        self.matrix_a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        self.matrix_b = [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
        
    def matrix_multiply(self, a, b):
        """3x3 행렬 곱셈 (순수 Python)"""
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    def format_matrix_result(self, matrix):
        """행렬을 Unreal Engine으로 전송할 형태로 포맷"""
        return "\n".join("[{:>4} {:>4} {:>4}]".format(*row) for row in matrix)
    
    def print_matrix(self, matrix, name):
        """행렬을 콘솔에 출력"""
        print(f"{name}:")
        for row in matrix:
            print(f"  {row}")
    
    def generate_random_matrix(self):
        """3x3 랜덤 행렬 생성"""
        return [[random.randint(1, 9) for _ in range(3)] for _ in range(3)]
        
    def start_server(self):
        """서버 시작"""
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            
            print(f"Matrix TCP Server listening on {self.host}:{self.port}")
            self.print_matrix(self.matrix_a, "Initial Matrix A")
            self.print_matrix(self.matrix_b, "Initial Matrix B")
            result = self.matrix_multiply(self.matrix_a, self.matrix_b)
            self.print_matrix(result, "Initial Result (A × B)")
            print("\nWaiting for Unreal Engine connection...")
            
            while self.running:
                try:
                    client, address = self.socket.accept()
                    self.clients.append(client)
                    print(f"✓ Unreal Engine connected from {address}")
                    
                    # 연결되면 즉시 행렬 곱셈 결과 전송
                    self.send_matrix_result(client)
                    
                    # 클라이언트 처리 스레드 시작
                    thread = threading.Thread(target=self.handle_client, args=(client,))
                    thread.daemon = True  # 메인 스레드 종료 시 함께 종료
                    thread.start()
                    
                except Exception as e:
                    if self.running:  # 서버가 종료 중이 아닐 때만 에러 출력
                        print(f"Connection error: {e}")
                    
        except Exception as e:
            print(f"Server startup error: {e}")
            print("Make sure:")
            print("1. Port 3500 is not being used by another program")
            print("2. No firewall is blocking the connection")
            print("3. Run this script before starting Unreal Engine")
    
    def handle_client(self, client):
        """클라이언트 연결 처리 및 주기적 데이터 전송"""
        count = 1
        while self.running and client in self.clients:
            try:
                time.sleep(5)  # 5초마다 새로운 계산 결과 전송
                
                # 새로운 랜덤 행렬 생성
                self.matrix_a = self.generate_random_matrix()
                self.matrix_b = self.generate_random_matrix()
                
                print(f"\n--- Calculation #{count} ---")
                self.send_matrix_result(client)
                count += 1
                
            except Exception as e:
                if self.running:  # 서버가 종료 중이 아닐 때만 에러 출력
                    print(f"Client handling error: {e}")
                break
        
        # 클라이언트 연결 정리
        self.disconnect_client(client)
    
    def send_matrix_result(self, client):
        """행렬 곱셈 결과만 Unreal Engine으로 전송"""
        try:
            # 행렬 곱셈 계산
            result_matrix = self.matrix_multiply(self.matrix_a, self.matrix_b)
            
            # UE4 TCP Socket Plugin에 맞는 간단한 문자열 형태
            message = self.format_matrix_result(result_matrix)
            
            # UTF-8로 인코딩
            encoded_message = message.encode("utf-8")
            
            # UE4 TCP Socket Plugin 형식: [4바이트 길이] + [메시지 본문]
            message_length = len(encoded_message)
            length_bytes = struct.pack('<I', message_length)  # 리틀 엔디언 (Little Endian)
            
            # 전송 (부분 송신 방지를 위해 sendall 사용)
            full_message = length_bytes + encoded_message
            client.sendall(full_message)
            
            # 서버 콘솔에 전송된 결과 표시
            print("✓ Sent matrix result to Unreal Engine:")
            print(f"Message: {message}")
            print(f"Length: {message_length} bytes")
            print(f"Full packet size: {len(full_message)} bytes")
            
        except Exception as e:
            print(f"Error sending matrix result: {e}")
            self.disconnect_client(client)
    
    def disconnect_client(self, client):
        """클라이언트 연결 해제"""
        try:
            client.close()
        except:
            pass
        
        if client in self.clients:
            self.clients.remove(client)
            print("✗ Client disconnected")
    
    def stop_server(self):
        """서버 종료"""
        print("\nStopping server...")
        self.running = False
        
        # 모든 클라이언트 연결 해제
        for client in self.clients[:]:  # 복사본으로 순회
            self.disconnect_client(client)
        
        # 서버 소켓 종료
        try:
            self.socket.close()
        except:
            pass

# 서버 실행
if __name__ == "__main__":
    print("=== Matrix Multiplication TCP Server ===")
    print("This server calculates 3x3 matrix multiplication")
    print("and sends ONLY the result matrix to Unreal Engine")
    print("Format: Clean matrix display without headers or metadata")
    print()
    
    server = MatrixTcpServer()
    try:
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        server.start_server()
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Shutting down server...")
        server.stop_server()
        print("Server stopped successfully!")
        print("Thanks for using Matrix TCP Server!")