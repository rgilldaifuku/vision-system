import cv2
import socket
import pickle
import struct

# Connect to the laptop's socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 🔴 REPLACE THIS WITH THE HOST IP PRINTED BY YOUR LAPTOP
laptop_ip = '10.20.232.188' 
port = 9999

client_socket.connect((laptop_ip, port))
data = b""
payload_size = struct.calcsize("Q")

while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet: break
        data += packet
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
        
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Deserialize the bytes back into an OpenCV image frame
    frame = pickle.loads(frame_data)
    
    # ----------------------------------------------------
    # YOUR VISION SOFTWARE / DETECTION LOGIC GOES HERE!
    # e.g., output = your_model.predict(frame)
    # ----------------------------------------------------
    
    # Display the frame to verify it's working
    cv2.imshow("Receiving Laptop Cam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()