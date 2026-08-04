#!/usr/bin/env python
import socket
import ast, re
import time

UDP_IP = "127.0.0.1" 
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Read the file line by line
with open("20241022_datastream.txt", "r") as file:
    for line in file:
        match = re.search(r"b'[^']*'", line)
        if match:
            byte_array_str = match.group(0)
            try:
                message = ast.literal_eval(byte_array_str)
                sock.sendto(message, (UDP_IP, UDP_PORT))
                print(f"Sent: {message}")
                time.sleep(0.02)
                
            except SyntaxError as e:
                print(f"Error while evaluating {byte_array_str}")

 
