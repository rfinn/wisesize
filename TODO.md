# TODO

## SGA-CIGALE

- [x] Compare the input photometry from the new SGA/AP03 table against the photometry Kim used, matched with the v2 VFID catalog.
- [x] Cross-match the WISEsize CIGALE results to NED-LVS by sky position and compare stellar mass and SFR, with redshift confirmation flags.
- [x] Cross-check overlapping WISEsize and Virgo Filament CIGALE results using VF positions and radial velocities.
- Rerun the 100-galaxy CIGALE comparison with the corrected v2 VFID-to-SGA2025 matches; the earlier run's southern half used coordinates from the wrong VF catalog version.
- Add one high-attenuation grid point back into the pruned model grid, e.g. `Av_ISM = 3.0`.
- Keep one long `tau_main` value to allow an approximately constant star-formation history, either restoring `tau_main = 1e5` or using `tau_main = 1e4`.
