import time
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit

kit = MotorKit(i2c=board.I2C(), address=0x60)

current_position = 0  # Starting position of the stepper motor


def move_stepper(target_position):
    global current_position
    steps_needed = target_position - current_position

    if steps_needed > 0:
        direction = stepper.BACKWARD
    else:
        direction = stepper.FORWARD

    # Move the stepper motor the required number of steps
    for _ in range(abs(steps_needed)):
        kit.stepper2.onestep(direction=direction, style=stepper.DOUBLE)
        time.sleep(0.01)

    # Update the current position
    current_position = target_position


def main():
    global current_position
    while True:
        position_input = input("Enter the target position (0-350, -1 to exit): ")
        try:
            target_position = int(position_input)
            if target_position == -1:
                break
            if not 0 <= target_position <= 350:
                print("Invalid position. Please enter a position between 0 and 350.")
                continue
            move_stepper(target_position)
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


if __name__ == "__main__":
    main()

kit.stepper2.release()
print("Stepper motor released.")
