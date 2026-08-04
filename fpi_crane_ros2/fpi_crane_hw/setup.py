from setuptools import find_packages, setup

package_name = "fpi_crane_hw"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/plc_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Heshan Fernando",
    maintainer_email="heshan.fernando@fpinnovations.ca",
    description="ROS 2 PLC UDP bridge node converted from ROS 1 rospy code.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "plc_node = fpi_crane_hw.plc_node:main",
            "plc_emulator = fpi_crane_hw.plc_emulator:main",
            "relative_joint_mover = fpi_crane_hw.relative_joint_mover:main",
            "keyboard_joy_node = fpi_crane_hw.keyboard_joy_node:main",
        ],
    },
)
