import numpy as np

def single_real_angle_to_cmd(real_angle): # 角度制
    target_angle = round(real_angle * 100)

    if (target_angle < 0) :
        target_angle += 65536

    return target_angle

def real_angle_to_cmd(real_angle): # 角度制
    if isinstance(real_angle, list):
        return [single_real_angle_to_cmd(angle) for angle in real_angle]
    elif isinstance(real_angle, np.ndarray):
        return np.array([single_real_angle_to_cmd(angle) for angle in real_angle])
    else:
        return single_real_angle_to_cmd(real_angle)

def real_radian_to_cmd(real_radian):
    if isinstance(real_radian, list):
        real_angle = [angle * 180 / np.pi for angle in real_radian]
    else: # for both scalar and np.ndarray
        real_angle = real_radian * 180 / np.pi
    return real_angle_to_cmd(real_angle)

def single_cmd_to_real_angle(cmd):
    if (cmd > 32767) :
        cmd -= 65535
    real_angle = cmd / 100.0
    return real_angle

def cmd_to_real_angle(cmd):
    if isinstance(cmd, list):
        return [single_cmd_to_real_angle(angle) for angle in cmd]
    elif isinstance(cmd, np.ndarray):
        return np.array([single_cmd_to_real_angle(angle) for angle in cmd])
    else:
        return single_cmd_to_real_angle(cmd)

def cmd_to_real_radian(cmd):
    real_angle = cmd_to_real_angle(cmd)
    if isinstance(real_angle, list):
        real_radian = [angle * np.pi / 180 for angle in real_angle]
    else: # for both scalar and np.ndarray
        real_radian = real_angle * np.pi / 180
    return real_radian