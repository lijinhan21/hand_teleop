import time

import init_path

from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient
from protocol.roh_registers_v1 import *

COM_PORT = '/dev/ttyUSB0'
NODE_ID = 2

# client = ModbusSerialClient(COM_PORT, FramerType.RTU, 115200)
client = ModbusSerialClient(
    port=COM_PORT,
    baudrate=115200,
    framer='rtu'    # 使用 framer 替代 method
)
client.connect()

if __name__ == "__main__":
    
    # 张开
    resp = client.write_registers(ROH_FINGER_POS_TARGET0, [0], slave=NODE_ID)
    time.sleep(2)
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 0], slave=NODE_ID)
    time.sleep(2)