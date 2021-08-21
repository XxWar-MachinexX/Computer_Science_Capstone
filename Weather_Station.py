# Jose F. Pina Jr.
# Southern New Hampshire University
# CS-350 Emerging Systems Architecture & Technology
# Final Project __Weather Station__

import grovepi
import math
import json
import time
import datetime
from grovepi import *

# LED configuration
green_led = 2
blue_led = 3
red_led = 4

# Temp / humidity sensor on port 5
sensor = 5

# Grove light sensor to analog port A0
light_sensor = 0

# Record data if light detected
threshold = 20
grovepi.pinMode(light_sensor, "INPUT")

# Sensor types
blue = 0
white = 1

# Initiate LEDs
pinMode(blue_led, "OUTPUT")
pinMode(green_led, "OUTPUT")
pinMode(red_led, "OUTPUT")

counter = 0.0
time_gap = 1800

# Function to control lights
def lights(red, green, blue):
    digitalWrite(red_led, red)
    digitalWrite(green_led, green)
    digitalWrite(blue_led, blue)

# Lights start off in the off position
lights(0,0,0)

# Creates JSON file to store data
with open("weather_station_data.json", "a") as write_file:
    write_file.write('[\n')
    
# Main
while (True & (counter < 11)):
    try:
        # First parameter is port, second parameter is sensor type
        [temp,humidity] = grovepi.dht(sensor, blue)

        # Celsius to Fahrenheit conversion
        temp = ((9 * temp) / 5) + 32

        # Output variables
        t = str(temp)
        h = str(humidity)

        # Creates data object to hold data
        data = [
            counter,
            temp,
            humidity
            ]

        # Sensor value
        sensor_value = grovepi.analogRead(light_sensor)

        # Calculate resistance of sensor in K
        resistance = (float)(1023 - sensor_value) * 10 / sensor_value

        print("sensor_value = %d resistance = %.2f" %(sensor_value, resistance))

        if (sensor_value > threshold):

            # Check validity of data
            if math.isnan(temp) == False and math.isnan(humidity) == False:

                # Prints output to screen
                print(counter)
                print("temp = %.02f F humidity = %.02f%%" %(temp, humidity))

                # Write output to JSON file
                with open("weather_station_data.json", "a") as write_file:
                    json.dump(data, write_file)
                    counter += 0.5
                    if (counter < 11):
                        write_file.write(',\n')

                # LED logic
                if (temp > 60 and temp < 85 and humidity < 80):
                    lights(0,0,1)

                elif (temp > 85 and temp < 95 and humidity < 80):
                    lights(0,1,0)

                elif (temp > 95):
                    lights(1,0,0)

                elif (humidity > 80):
                    lights(0,1,1)

                else:
                    lights(0,0,0)

                time.sleep(time_gap)

        else:
            print("Data not recorded, sensor can not detect light")
            time,sleep(10)

    # Catch exception for dividing by zero
    except ZeroDivisionError:
        print("Zero reading on sensor")

    except KeyboardInterrupt:
        lights(0,0,0)

    except IOError:
        print("ERROR")

print("Data Recorded")
lights(0,0,0)

# Closing bracket on JSON file
with open("weather_station_data.json", "a") as write_file:
    write_file.write('\n]')
    

            








                




















        














