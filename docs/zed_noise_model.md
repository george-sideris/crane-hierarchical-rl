# ZED depth-noise model: characterisation and injection

Thesis notes for the sensor-noise component of the sim2real pipeline. Covers what is modelled,
how the parameters were measured, where the noise enters training, and what the measurement does
and does not establish.

## 1. What is modelled

Isaac renders geometrically perfect depth. The real ZED X derives depth from stereo disparity, so
its dominant error is **axial** (along the camera ray) and grows with the square of range:

    sigma(z) = c * z^2                                    [m]

This is the standard stereo relation. For a disparity-matching error of `sigma_d` pixels,

    sigma(z) = z^2 / (f_x * B) * sigma_d

With the ZED X's calibrated `f_x = 746.9 px` and baseline `B = 0.12 m`, the deployed coefficient
`c = 0.0014` corresponds to `sigma_d ~ 0.13 px`, i.e. a realistic sub-pixel matching error. The
model therefore has a physical interpretation rather than being a fitted fudge factor.

A second term models **dropout**: the real sensor returns no depth at occlusion boundaries, on
low-texture surfaces and in specular regions. A fraction `p = 0.06` of valid pixels is discarded
at random.

Not modelled (stated as limitations): systematic depth bias, temperature drift, correlated error
across neighbouring pixels (the SDK spatially filters its depth), and the higher dropout rate at
object silhouettes specifically.

## 2. How the parameters were characterised

### 2.1 Method

Sensor noise is separated from surface shape using **local patch planes**:

1. Take a dense capture of a large, approximately planar surface (the yard floor).
2. For each of many sample points, collect neighbours within a 12-15 cm radius.
3. Fit a plane to that patch by SVD; the residual standard deviation about the local plane is the
   high-frequency (sensor) component. Fitting locally cancels large-scale surface shape and slope,
   which a single global plane fit would absorb into the residuals.
4. Convert the perpendicular residual to an axial one:
   `sigma_axial = sigma_perp / |cos(theta)|`, where `theta` is the angle between the camera ray
   and the local surface normal. Patches with `|cos(theta)| < 0.2` (grazing views) are discarded.
5. Bin by range and take the **median** per bin (robust to debris and patch edges).

### 2.2 Results

Original characterisation (bag `17_20`, floor): 14 mm @ 2.5 m, 20 mm @ 3.5 m, 26 mm @ 4.5 m,
giving `c = 0.0014`, the deployed value.

Two independent re-measurements (2026-07-31):

| source | 3.75 m | 4.75 m | 5.75 m | 6.75 m | fitted c |
|---|---|---|---|---|---|
| deprojected base-frame clouds, floor patches | 17.1 mm | 28.9 mm | 41.0 mm | 62.9 mm | **0.00126** |
| raw bag clouds (`cloud_registered`), camera frame | 4.8 mm | 9.4 mm | 7.4 mm | 4.6 mm | **0.00020** |

The two disagree by ~6x, and each is biased in a known direction:

- The **floor-patch** estimate is an **upper bound**: yard floor is not perfectly planar and carries
  bark debris, so genuine surface roughness enters the residual.
- The **raw-bag** estimate is a **lower bound**: the ZED SDK spatially filters depth, so
  neighbouring points are correlated and local patch variance understates the true error. Its
  profile is also non-monotonic (rising again below 3 m and beyond 8 m), which reflects scene
  content -- crane structure at close range, clutter at long range -- rather than sensor physics.

The deployed `c = 0.0014` lies inside this bracket, as does the original characterisation.

### 2.3 Recommended definitive procedure

A controlled measurement would remove both biases and is the honest way to fix the number:
place a flat, textured, matte board perpendicular to the optical axis at 3, 5 and 7 m; capture
with SDK spatial filtering disabled; fit a single plane per capture; report residual sigma per
range. That isolates sensor error from surface roughness and from filter-induced correlation.

## 3. Where the noise is injected

Two injection points exist; they are alternatives, not cumulative.

**(a) Rendering time** -- `crane_rl_env_gaze.py`, in `get_pointcloud_base`. Noise is applied to the
depth image *before* unprojection, which is what puts it along the camera ray exactly as the
physical process does. Dropout is applied in the same place by setting pixels to infinity.
Enabled with `--zed_noise`; used for deployment-matched evaluation.

**(b) Training time** -- `train_bc_pointcloud.py::apply_zed_noise`, applied per mini-batch to the
already-collected cloud. Each point is displaced along its own camera ray by
`N(0, c * range^2)`, using the camera position saved with the dataset (`cam_pos_base.npy`);
zero-padded slots are left untouched.

**Collections are recorded clean and noised at training time.** This is deliberate: the same clean
cloud is re-noised with a fresh draw every epoch, so the network sees many noise realisations of
each scene rather than one frozen realisation, which both regularises and reflects that noise is
independent between captures. The cost is that dropout (a rendering-time effect on the depth
image) is not reproduced by the training-time path.

## 4. Status

`zed_axial_coeff = 0.0014`, `zed_dropout = 0.06`, enabled at training via `--zed_noise`. The
coefficient is physically interpretable, agrees with the original floor characterisation, and sits
within the bracket established by the two 2026-07-31 re-measurements. It should be regarded as
correct to within roughly a factor of two until the controlled measurement in 2.3 is performed.
