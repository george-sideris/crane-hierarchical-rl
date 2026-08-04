from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ikpy.chain import Chain
from sensor_msgs.msg import JointState


@dataclass(frozen=True)
class FKResult:
    transform: np.ndarray
    z_axis: np.ndarray


class CraneFK:
    """
    Forward-kinematics model for:

        base_link -> grapplecarrier

    Joint positions are taken from sensor_msgs/JointState by joint name.
    """

    REQUIRED_JOINTS = (
        "slew_joint",
        "boom_joint",
        "stick_joint",
        "telescope_joint",
        "hanger_joint",
        "bearingfork_joint",
        "grapplecarrier_joint",
    )

    def __init__(self, urdf_path: str):
        # Giving the complete path avoids IKPy accidentally following a
        # different branch after grapplecarrier.
        #
        # Verify these link names against the actual URDF. Your older class
        # used "basemast" as its starting element, so this may need adjustment.
        chain_elements = [
            "base_link",
            "slew_joint",
            "mast",
            "boom_joint",
            "mainboom",
            "stick_joint",
            "stick",
            "telescope_joint",
            "telescope",
            "hanger_joint",
            "upperpassive",
            "bearingfork_joint",
            "lowerpassive",
            "grapplecarrier_joint",
            "grapplecarrier",
        ]

        active_mask = [False, True, True, True, True, True, True, True]

        self.chain = Chain.from_urdf_file(
            urdf_file=urdf_path,
            active_links_mask=active_mask,
            base_elements=["base_link"],
        )

        self._chain_joint_names = [
            link.name for link in self.chain.links
        ]

        self._validate_chain()

    def _validate_chain(self) -> None:
        """
        Fail immediately if the expected crane joints are not in the IKPy
        chain. This catches incorrect URDF path names during startup.
        """
        missing = [
            name
            for name in self.REQUIRED_JOINTS
            if name not in self._chain_joint_names
        ]

        if missing:
            raise RuntimeError(
                "IKPy chain does not contain required joints: "
                f"{missing}. Chain contains: {self._chain_joint_names}"
            )

    @staticmethod
    def joint_state_to_dict(msg: JointState) -> dict[str, float]:
        if len(msg.name) != len(msg.position):
            raise ValueError(
                "JointState name and position arrays have different lengths: "
                f"{len(msg.name)} vs {len(msg.position)}"
            )

        return {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
        }

    def make_ikpy_joint_vector(
        self,
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        """
        Construct the full joint vector expected by IKPy.

        IKPy requires a value for every element in its chain, including
        inactive and fixed elements. Fixed/origin elements remain zero.
        """
        missing = [
            name
            for name in self.REQUIRED_JOINTS
            if name not in joint_positions
        ]

        if missing:
            raise ValueError(
                f"JointState is missing required joints: {missing}"
            )

        q_full = np.zeros(len(self.chain.links), dtype=float)

        for index, chain_link in enumerate(self.chain.links):
            joint_name = chain_link.name

            if joint_name in joint_positions:
                q_full[index] = joint_positions[joint_name]

        return q_full

    def forward_from_joint_state(self, msg: JointState) -> FKResult:
        joint_positions = self.joint_state_to_dict(msg)
        q_full = self.make_ikpy_joint_vector(joint_positions)

        T_base_grapple = np.asarray(
            self.chain.forward_kinematics(q_full),
            dtype=float,
        )

        if T_base_grapple.shape != (4, 4):
            raise RuntimeError(
                "IKPy returned an unexpected FK matrix shape: "
                f"{T_base_grapple.shape}"
            )

        R_base_grapple = T_base_grapple[:3, :3]

        # Third column is grapplecarrier +z expressed in base_link.
        z_axis = R_base_grapple[:, 2]
        z_norm = np.linalg.norm(z_axis)

        if z_norm < 1e-12:
            raise RuntimeError("Computed grapple z-axis has zero magnitude")

        z_axis = z_axis / z_norm

        return FKResult(
            transform=T_base_grapple,
            z_axis=z_axis,
        )