# ApplePro2 SpatialDSP (HeSuVi `Apple_Pro2_Stereo.wav` -> iOS `.dylib`)

Converts the 14-channel HeSuVi HRIR (`Apple_Pro2_Stereo.wav`) into a 4-channel True-Stereo (`LL, LR, RL, RR`) real-time `vDSP_conv` processor for iOS (`44.1 kHz` & `48 kHz`).

- **`SpatialDSP_ApplePro2_Matrix.dylib`**: Full 14-channel HeSuVi Stereo Matrix Upmix folded into True-Stereo.
- **`SpatialDSP_ApplePro2_Front.dylib`**: Pure 1:1 Front Left/Right (`FL/FR`) True-Stereo.
