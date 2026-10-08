import board
import busio
from adafruit_servokit import ServoKit
from time import sleep

# 正確初始化 I2C（會自動用正確 bus）
i2c = busio.I2C(board.SCL, board.SDA)

kit = ServoKit(channels=16, i2c=i2c)

servo = kit.servo[0]
servo.set_pulse_width_range(500, 2500)

print("Testing servo...")

try:
    while True:
        servo.angle = 0
        sleep(1)

        servo.angle = 90
        sleep(1)

        servo.angle = 180
        sleep(1)

except KeyboardInterrupt:
    pass
