### if after running app.py you geet this error:
<pre style="color:red;">
Traceback (most recent call last):
  File "/home/timer/hand-pinch-gesture-timer/app.py", line 37, in <module>
    GPIO.setup(SERVO_PIN, GPIO.OUT)
RuntimeError: No access to /dev/mem.  Try running as root!
</pre>

### try this:
<pre>
sudo groupadd gpio
sudo usermod -a -G gpio user_name
sudo grep gpio /etc/group
sudo chown root.gpio /dev/gpiomem
sudo chmod g+rw /dev/gpiomem
</pre>

### other useful commands:
<pre>
scp pi@192.***.*.***:/home/pi/example.txt ~/Desktop/

scp timer@timer:~/testoo.jpg ~/Documents
</pre>

### other useful resources:
- https://pyimagesearch.com/2016/08/29/common-errors-using-the-raspberry-pi-camera-module/
- https://pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

