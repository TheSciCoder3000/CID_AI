import argparse
import serial
import serial.tools.list_ports
import time
from typing import List

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def port_identifier() -> List[str]:
    """Retruns a List of PORTS available"""
    ports = serial.tools.list_ports.comports()
    portsList = {}
    print("---------------------- ALL AVAILABLE COM PORTS ----------------------")
    print(" ")
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
        portsList[port] = desc
    return portsList

def getSerial(COM: str, coms: List[str]):
    """Returns a serial object from `COM` input and list of `coms`"""
    if COM in coms:
        ser = serial.Serial(COM, 9600)
        time.sleep(5)
        return ser