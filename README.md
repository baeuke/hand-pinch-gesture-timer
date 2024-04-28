###if after running app.py you geet this error:
<pre style="color:red;">
Traceback (most recent call last):
  File "/home/timer/hand-pinch-gesture-timer/app.py", line 37, in <module>
    GPIO.setup(SERVO_PIN, GPIO.OUT)
RuntimeError: No access to /dev/mem.  Try running as root!
</pre>

###try this:
<pre>
sudo groupadd gpio
sudo usermod -a -G gpio user_name
sudo grep gpio /etc/group
sudo chown root.gpio /dev/gpiomem
sudo chmod g+rw /dev/gpiomem
</pre>