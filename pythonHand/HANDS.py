import cv2
import numpy as np
import mediapipe as mp
import socket
import win32api

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 12345))
server_socket.listen(1)
connection, address = server_socket.accept()
print("Connected by", address)

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    prev_index_finger_tip_x, prev_index_finger_tip_y = 0, 0

    sensitivity_factor = 5
    smoothing_factor = 0.5

    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                index_finger_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x, index_finger_tip_y = int(index_finger_tip.x * image_width), int(index_finger_tip.y * image_height)

                # чуствительность
                index_finger_tip_x *= sensitivity_factor
                index_finger_tip_y *= sensitivity_factor

                # плавнсть
                index_finger_tip_x = int((1 - smoothing_factor) * index_finger_tip_x + smoothing_factor * prev_index_finger_tip_x)
                index_finger_tip_y = int((1 - smoothing_factor) * index_finger_tip_y + smoothing_factor * prev_index_finger_tip_y)

                win32api.SetCursorPos((index_finger_tip_x, index_finger_tip_y))


                prev_index_finger_tip_x, prev_index_finger_tip_y = index_finger_tip_x, index_finger_tip_y

                # Send hand position to Unity
                message = f"{index_finger_tip_x},{index_finger_tip_y}"
                connection.send(message.encode())

        cv2.imshow('game', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


server_socket.close()
cv2.destroyAllWindows()
