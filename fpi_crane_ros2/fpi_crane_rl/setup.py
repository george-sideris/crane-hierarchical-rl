from setuptools import find_packages, setup

package_name = "fpi_crane_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/checkpoints", ["checkpoints/model_350.pt"]),
        ("share/" + package_name + "/urdf", ["urdf/fpiforwarder-upperpassive.urdf"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="George Sideris",
    maintainer_email="georgesiderisdagres@gmail.com",
    description="Learned point-cloud grasping policy for the crane lab: PointNet policy, pick-and-place FSM, and trajectory bridge to the PLC node.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "crane_policy_node = fpi_crane_rl.crane_policy_node:main",
            "test_target = fpi_crane_rl.test_target:main",
            "send_gaze_pose = fpi_crane_rl.send_gaze_pose:main",
            "lift = fpi_crane_rl.lift:main",
            "characterize = fpi_crane_rl.characterize:main",
        ],
    },
)
