import time
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit


kit = MotorKit(i2c=board.I2C(), address=0x60)

def move_stepper(steps, direction):
    if direction == "forward":
        for _ in range(abs(steps)):
            kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE)
            time.sleep(0.01)
    elif direction == "backward":
        for _ in range(abs(steps)):
            kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
            time.sleep(0.01)

def main():
    while True:
        steps_input = input("Enter the number of steps (+/- to indicate direction, 0 to exit): ")
        try:
            steps = int(steps_input)
            if steps == 0:
                break
            move_stepper(steps, "backward" if steps > 0 else "forward")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    main()

kit.stepper2.release()
print("Stepper motor released.")
