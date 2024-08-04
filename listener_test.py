from pynput import keyboard

def on_press(key):
    try:
        if key.char == 'w':
            print("Move forward")
        elif key.char == 'a':
            print("Move left")
        elif key.char == 's':
            print("Move back")
        elif key.char == 'd':
            print("Move right")
        elif key.char == 't':
            print("Take off")
        elif key.char == 'l':
            print("Land")
    except AttributeError:
        if key == keyboard.Key.up:
            print("Move up")
        elif key == keyboard.Key.down:
            print("Move down")
        elif key == keyboard.Key.left:
            print("Rotate counter clockwise")
        elif key == keyboard.Key.right:
            print("Rotate clockwise")
        elif key == keyboard.Key.esc:
            return False

# Start the listener thread for keyboard input
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Keep the program running to listen to key presses
listener.join()