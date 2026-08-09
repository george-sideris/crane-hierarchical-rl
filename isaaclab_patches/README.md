# IsaacLab-side local changes

The crane work lives in this repo, but the container build and one IsaacLab
package file are edited inside the IsaacLab checkout itself. That checkout's
only remote is upstream `git@github.com:isaac-sim/IsaacLab.git`, so those edits
cannot be pushed anywhere and would otherwise not follow this repo to another
machine. They are exported here instead.

`isaaclab_local_changes.patch` covers:

- `docker/Dockerfile.base`
- `docker/Dockerfile.ros2`
- `source/isaaclab/setup.py`

Base commit the patch was taken against: `4f81564f3cd2266459f45a35154cd85eaa3d9b4b`
("added SAC training").

`_structures.py` is the file `Dockerfile.base` COPYs into Isaac Sim's prebundled torch. The
patch only contains the COPY line, so without this file beside it the docker build fails at
that step with `COPY failed: file not found`. It must be placed at `docker/patches/` inside
the IsaacLab checkout BEFORE building (the apply step below does this).

## Applying on another machine

Clone IsaacLab, check out the base commit (or a nearby one), put this repo at
`IsaacLab/crane_testbed`, then from the IsaacLab root:

```bash
mkdir -p docker/patches
cp crane_testbed/isaaclab_patches/_structures.py docker/patches/_structures.py
git apply --check crane_testbed/isaaclab_patches/isaaclab_local_changes.patch   # dry run
git apply crane_testbed/isaaclab_patches/isaaclab_local_changes.patch
```

`crane_testbed/cloud_setup.sh setup` does all of the above unattended; see that script for a
rented-GPU walkthrough (including which container profile you need - sim-only training wants
`base`, NOT `ros2`).

If `--check` fails because upstream has moved, apply with three-way merge:

```bash
git apply -3 crane_testbed/isaaclab_patches/isaaclab_local_changes.patch
```

## Regenerating this patch

From the IsaacLab root, after further edits to those files:

```bash
git diff -- docker/Dockerfile.base docker/Dockerfile.ros2 source/isaaclab/setup.py \
  > crane_testbed/isaaclab_patches/isaaclab_local_changes.patch
```

Keep the base commit line above in sync when you do.
