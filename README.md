# Quadplane Transition Corridor

Modelling and control of the hover-to-cruise transition on a quadplane, with
statistical characterisation of where and why the controller departs.

Abdou Wade · Aerospace Engineering (ACCEND) · University of Cincinnati

## The problem

A quadplane has two personalities. In hover, four lift rotors hold it up and
the wing does nothing. In cruise, a pusher prop drives it forward, the wing
carries the weight, and the lift rotors are dead weight.

Between them is a window of roughly eight seconds where the aircraft is too
slow to fly and too fast to hover. Control power scales with dynamic pressure,

    q = 0.5 * rho * V^2

so at 5 m/s an elevon delivers about 8% of its cruise authority. That window is
the transition corridor.

## Deliverable

Not "I made a transition work in simulation." The deliverable is:

- a gain-scheduled controller that crosses the corridor,
- the precise conditions under which it departs,
- a nonlinear (INDI) controller that survives those conditions,
- and the difference proven across ~1000 randomised runs with parameter
  distributions justified by system identification, not invented.

## Layout

    control/      PID, cascaded loops, LQR          (L0, L1, L3)
    dynamics/     rotations, 6-DOF equations of motion (L0, L1)
    estimation/   complementary filter, EKF, noise ID  (L2, L4)
    aero/         lift, drag, stall, hysteresis        (L5)
    transition/   corridor mapping, blending           (L7)
    allocation/   pseudo-inverse, QP, INDI             (L6)
    sim/          PX4 bridge, Monte Carlo              (L8)
    analysis/     NEES/NIS consistency, plots          (L4+)
    prob/         Gaussians, covariance, van Loan      (Module 01)
    linalg/       NumPy implementations                (all term)
    cpp/          Eigen port of hot paths              (Outro)
    golden/       reference trajectories for C++ regression
    tests/        pytest, from day one

## Running

    source .venv/bin/activate
    pytest                  # 5 passed
    python -m control.pid   # writes results/week01_pid_step.png

## Status

- [x] Module 01 — environment, toolchain, first commanded flight
- [ ] Level 0 — PID, rotations, sensor models
- [ ] Level 1 — multirotor cascade
- [ ] Level 2 — sensor fusion
- [ ] Level 3 — state space, trim, LQR
- [ ] Level 4 — Kalman filter, consistency, noise ID
- [ ] Level 5 — nonlinear aerodynamics, stall
- [ ] Level 6 — control allocation, INDI
- [ ] Level 7 — the corridor, and breaking it
- [ ] Level 8 — system ID, Monte Carlo campaign
