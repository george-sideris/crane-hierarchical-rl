import pyzed.sl as sl

cam = sl.Camera()
status = cam.open()
print(f"Camera open status: {status}")
if status == sl.ERROR_CODE.SUCCESS:
    info = cam.get_camera_information()
    print(f"Model: {info.camera_model}")
    print(f"Serial: {info.serial_number}")
    cam.close()
