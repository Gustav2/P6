"""
config/satellite.py — Satellite orbit, constellation, and handover parameters
"""

SAT_HEIGHT_M = 550_000.0
"""
Nominal LEO satellite orbital altitude [m].
- 550 km is the operational altitude of SpaceX Starlink Shell 1 (the primary
  deployment shell as of 2023–2024).  Earlier filings used 600 km but
  Starlink transitioned to 550 km (FCC approval 2019, operational 2020–).
- One-way propagation delay at 550 km nadir: ~1.83 ms (c = 3×10⁸ m/s).
- Free-space path loss at 3.5 GHz, 550 km slant: ~180 dB.
Source: [Starlink-FCC] Attachment A, Table 1 (550 km shell, 53° incl.).
         Also [3GPP-38.821] §6.1 uses 600 km as the 3GPP reference; 550 km
         is the closest real-world analogue.
"""

ELEVATION_ANGLE_DEG = 60.0
"""
Minimum UE-to-satellite elevation angle [degrees] used for the PHY
channel model constructor.  The RT simulation sweeps through multiple
elevation angles as the constellation moves overhead.
"""

SAT_ORBIT_INCLINATION_DEG = 53.0
"""
Orbital inclination of the satellite constellation [degrees].
- 53° matches SpaceX Starlink Shell 1 inclination (FCC filing 2019).
- Affects the ground track and how quickly satellites pass overhead.
Source: [Starlink-FCC] Attachment A.
"""

SAT_ORBITAL_VELOCITY_MS = 7612.0
"""
Orbital velocity of a LEO satellite at SAT_HEIGHT_M [m/s].
- Derived from the vis-viva equation: v = sqrt(GM / (R_E + h))
    GM = 3.986004418×10¹⁴ m³/s²  (WGS84 standard gravitational parameter)
    R_E = 6.3781×10⁶ m            (WGS84 equatorial radius)
    h   = 5.50×10⁵ m              (550 km altitude)
  → v = sqrt(3.986e14 / 6.928e6) = 7,612 m/s
- Doppler shift at 2.0 GHz (worst case, near horizon where radial velocity peaks):
    Δf_max = v/c × f = 7612 / 3×10⁸ × 2.0×10⁹ ≈ ±50.7 kHz (full radial pass)
    Δf at zenith ≈ 0 (perpendicular to LOS); peak near horizon ≈ ±50.7 kHz.
  At µ=1 (30 kHz SCS) this is ~1.7× SCS — within pre-compensation range.
  At µ=0 (15 kHz SCS) it would be ~3.4× SCS, causing significant ICI.
- Used in ntn_phy.py: ut_velocities is set to zero, NOT to this value.
  3GPP TR 38.821 §6.1.2 mandates that NTN UEs pre-compensate the satellite
  Doppler shift before the OFDM demodulator.  Setting ut_velocities = 0
  models the post-compensation residual.  This constant is retained for
  documentation and for the NTN Doppler branch in Sionna (bs_height ≥ 600 km).
Source: Standard orbital mechanics (IERS Conventions 2010, §6.1).
"""

CONSTELLATION_TOTAL_SATS = 3000
"""
Approximate total number of satellites in the reference LEO shell.

- This is metadata used to document constellation realism; the simulator does
  not instantiate all of these satellites in NS-3 or Sionna RT.
- 3000 is in-family with modern broadband LEO shell sizes (order of 10^3).
"""

VISIBLE_SATELLITES_PER_PASS = 8
"""
Number of satellites sampled as simultaneously visible during one urban pass.

- This is the practical simulation density used by ray tracing and handover
  scheduling (NUM_SATELLITES below).
- 8 provides a more realistic visible-satellite density than 3 while keeping
  runtime manageable on CPU-only setups.
"""

NUM_SATELLITES = VISIBLE_SATELLITES_PER_PASS
"""
Number of satellites in the simulated constellation pass.

This controls both the Sionna RT ray-tracing snapshot and the NS-3 handover
schedule. It is interpreted as the sampled set of currently visible satellites
from a much larger LEO constellation (CONSTELLATION_TOTAL_SATS).

Use this value to trade realism vs runtime:
  - Higher values improve handover granularity and elevation diversity.
  - Lower values run faster.
"""

SAT_HANDOVER_ELEVATION_DEG = 10.0
"""
Elevation angle threshold [degrees] at which a handover is triggered.
- When the serving satellite falls below this angle, the UE connects
  to the next satellite that has risen above this threshold.
- 10° is the minimum elevation specified for LEO NTN handover in the
  3GPP NTN reference scenario.
Source: [3GPP-38.821] §6.1, Table 6.1.1-1 (minimum elevation angle = 10°).
"""

SAT_SPACING_DEG = 15.0
"""
Angular spacing between consecutive satellites along the orbit [degrees].
- Determines how long the gap between satellites is (and therefore
  whether seamless handover is possible or a link interruption occurs).
- At 550 km altitude, 15° ≈ 144 km ground-track separation.
"""
