# WEATHER STATION
# Created by Jose Pina Jr.

# BME280 SENSOR
# The bme280 sensor is a device that measures temperature, humidity, and pressure
# Install the bme280 sensor libraries ==> 'sudo pip install pimoroni-bme280 smbus'
# Install openpyxl libraries ==> 'sudo pip install openpyxl
# Enable I2C
# Enable SSH

# ANEMOMETER
# An anemometer is a device that measures windspeed

# PARTS NEEDED
# Raspberry Pi
# IR Reflectance Sensor
# BME280 Sensor
# 2 ping pong balls
# 10 inch cylindrical shaft
# Nuts and Bolts 4x1/2"
# Box cutter
# Liquid poly adhesive
# 2mm HIPS sheet also known as plasticard
# 470 ohm resistor
# 1k ohm resistor
# 10k ohm resistor
# Wire/connectors
# PCB
# Soldering Gun
# Solder

# DESCRIPTION
# Cut the ping pong balls in half to make 4 cups. Cut 4 strips of the 2mm HIPS sheet about 5 inches in
# Length and 1/2 inch wide. Drill holes on one side of the strips about 1/2 an inch from the end. Drill
# A hole in the middle of the ping pong ball halves. Attach the ping pong balls to the HIPS strips and
# Secure with nuts and bolts. This should resemble a makeshift spoon. Use the liquid poly adhesive to
# Secure opposite ends of strips to top of cylindrical shaft so that they are 90 degrees of each other.
# If viewed from the top of the shaft, this should look like a plus sign. Cut 5 evenly sized squares and
# Using the liquid adhesive create a box. The bottom will remain open. Drill a hole on the top center
# Square that allows the shaft to slide through and spin freely. Cut a disc out of the HIPS sheet that
# Fits inside the box. Drill a hole in the center so that the shaft can slide through. If your disc is not black,
# Paint it black now. Then paint an 1/8 of the disc white. Attach the disc to the cylindrical shaft using
# Liquid adhesive. Attach IR Reflectance Sensor to inside of the box with liquid adhesive so that the
# IR sensors are facing the disc. Attach the 470 ohm resistor to the positive lead of the IR Reflectance
# Sensor. Attach other end of 470 ohm resistor to GPIO 2. Attach positive lead of the IR sensor to
# The 10k ohm resistor and attach both to GPIO 6. Attach other end of 10k ohm resistor to the 1k ohm
# Resistor and attach both to the negative lead on the IR sensor. Attach other end of 1k ohm resistor to
# GPIO 12. Also attach negative lead of IR sensor to GPIO 17.

# CODE
# Import libraries
from smbus import SMBus
from bme280 import BME280
import time
import datetime
from datetime import date
from openpyxl import load_workbook
import RPi.GPIO as GPIO

# Set up GPIO for wind speed
GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.IN)

# Anemometer arm diameter in millimeters (from center of cup to center of cup)
arm_diameter = float(106)

# Calculate arm circumference in meters
arm_circ = float(arm_diameter/1000)*3.1415

# Set an anemometer factor to account for inefficiency (value is guessed)
afactor = float(2.5)

# Initialize the BME280
bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)

# Disgarding the first reading
temperature = bme280.get_temperature()
pressure = bme280.get_pressure()
humidity = bme280.get_humidity()
time.sleep(1)

# Load workbook
wb = load_workbook('/home/pi/Python_Code/weather.xlsx')
sheet = wb[ 'Sheet1' ]

try:
    while True:

        # Read the sensor and get date and time
        temperature = round(bme280.get_temperature(),1)
        pressure = round(bme280.get_pressure(),1)
        humidity = round(bme280.get_humidity(),1)
        today = date.today()
        now = datetime.datetime.now().time()

        # Measure wind speed
        rotations = float(0)
        trigger = 0
        end_time = time.time() + 10
        sensor_start = GPIO.input(12)
        while time.time() < end_time:
            if GPIO.input(12) == 1 and trigger == 0:
                rotations += 1
                trigger = 1
            if GPIO.input(12) == 0:
                trigger = 0
            time.sleep(0.001)

        if rotations == 1 and sensor_start == 1:
            rotations = 0
        rots_per_second = float(rotations/10)
        windspeed = float((rots_per_second) * arm_circ * afactor)

        # Add data to spreadsheet
        print('Adding data to spreadsheet')
        print('{:.1f}*C {} hPa {}% {:.2f} m/s'.format(temperature, pressure, humidity, windspeed))
        row = (today, now, temperature, pressure, humidity, windspeed)
        sheet.append(row)

        #Save the workbook
        wb.save('/home/pi/Python_Code/weather.xlsx')

        # Wait for 10 minutes (600 seconds)
        time.sleep(600)

finally:
    # Ensure workbook is saved
    wb.save('/home/pi/Python_Code/weather.xlsx')

    print('Goodbye!')
        
