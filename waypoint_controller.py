"""
waypoint_controller.py  (v3 — corrected)
=========================================
Research-grade controller for VRX WAM-V.

Fixes vs v2
-----------
1. _unicycle STRAIGHT: differential correction was inverted (steered away
   from target). Fixed: left += corr, right -= corr so positive heading error
   (target to the left) increases left thrust and decreases right thrust.

2. LAWN_LEGS: pattern was folding back on itself (y went 0→12→24→36→24→12).
   Fixed to a proper lawnmower: y = 0, 12, 24, 36, 48, 60 with alternating
   x direction each pass.

3. main(): added rclpy already-initialized guard to prevent double-init crash.

Motion patterns
---------------
UNICYCLE  →  lawnmower / back-and-forth straight lines with sharp differential
             turns at each end.  Produces alternating surge + yaw-rate bursts
             that are characteristic of differential-drive unicycle kinematics.

BICYCLE   →  smooth circle using azimuth angle steering.  Produces constant
             curvature with low yaw-rate variation — characteristic of bicycle
             / car-like steering.

Usage
-----
    python3 waypoint_controller.py --mode unicycle
    python3 waypoint_controller.py --mode bicycle
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64

import numpy as np
import math
import argparse

EARTH_RADIUS        = 6371000.0
BOUNDARY_RADIUS     = 80.0
MAX_THRUST          =  1500.0
MIN_THRUST          = -1500.0

# ── FIX #2 ────────────────────────────────────────────────────────────────────
# Proper lawnmower: each leg advances y by LANE_WIDTH; x alternates direction.
# Pattern: →, ←, →, ←, →, ← (6 passes, 12 waypoints)
LANE_WIDTH = 12.0
LAWN_LEGS = [
    ( 10.0,  55.0,   0.0),   # pass 1 →
    ( 55.0,  10.0,  12.0),   # pass 2 ←
    ( 10.0,  55.0,  24.0),   # pass 3 →
    ( 55.0,  10.0,  36.0),   # pass 4 ←
    ( 10.0,  55.0,  48.0),   # pass 5 →   ← was incorrectly y=24 in v2
    ( 55.0,  10.0,  60.0),   # pass 6 ←   ← was incorrectly y=12 in v2
]
# ──────────────────────────────────────────────────────────────────────────────

CIRCLE_RADIUS       = 45.0
CIRCLE_CENTER       = np.array([0.0, 0.0])
N_CIRCLE_WPS        = 16

WP_ACCEPT_UNI       = 6.0
WP_ACCEPT_BIC       = 8.0
KP_HEADING_UNI      = 3.0
KP_HEADING_BIC      = 2.5
UNI_STRAIGHT_THRUST = 700.0
UNI_TURN_THRUST     = 600.0
BIC_CRUISE_THRUST   = 600.0


def wrap(a):
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def quat_to_yaw(q):
    """Extract yaw from a ROS quaternion message."""
    s = 2.0 * (q.w * q.z + q.x * q.y)
    c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(s, c)

def latlon_to_xy(lat, lon, olat, olon):
    """Convert lat/lon to local ENU metres relative to origin."""
    dlat = math.radians(lat - olat)
    dlon = math.radians(lon - olon)
    x = EARTH_RADIUS * dlon * math.cos(math.radians(olat))
    y = EARTH_RADIUS * dlat
    return x, y

def make_lawnmower_waypoints():
    wps = []
    for x_s, x_e, y in LAWN_LEGS:
        wps.append(np.array([x_s, y]))
        wps.append(np.array([x_e, y]))
    return wps

def make_circle_waypoints():
    wps = []
    for i in range(N_CIRCLE_WPS):
        a = 2.0 * math.pi * i / N_CIRCLE_WPS
        wps.append(CIRCLE_CENTER + CIRCLE_RADIUS * np.array([math.cos(a), math.sin(a)]))
    return wps


class WaypointController(Node):

    def __init__(self, mode='unicycle'):
        super().__init__('waypoint_controller')

        self.mode       = mode
        self.x          = 0.0
        self.y          = 0.0
        self.heading    = 0.0
        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.emergency  = False

        if mode == 'unicycle':
            self.waypoints = make_lawnmower_waypoints()
            self.wp_accept = WP_ACCEPT_UNI
        else:
            self.waypoints = make_circle_waypoints()
            self.wp_accept = WP_ACCEPT_BIC

        self.wp_index  = 0
        self.laps      = 0
        self.uni_state = 'align'   # unicycle state machine: 'align' | 'straight'

        self.get_logger().info(
            f"[{mode.upper()}] ready | {len(self.waypoints)} waypoints")

        self.pub_l = self.create_publisher(Float64, '/wamv/thrusters/left/thrust',  10)
        self.pub_r = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        if mode == 'bicycle':
            self.pub_lp = self.create_publisher(Float64, '/wamv/thrusters/left/pos',  10)
            self.pub_rp = self.create_publisher(Float64, '/wamv/thrusters/right/pos', 10)

        self.create_subscription(NavSatFix,
            '/wamv/sensors/gps/gps/fix', self.gps_cb, 10)
        self.create_subscription(Imu,
            '/wamv/sensors/imu/imu/data', self.imu_cb, 10)
        self.create_timer(0.1, self.control_loop)

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def imu_cb(self, msg):
        self.heading = quat_to_yaw(msg.orientation)

    def gps_cb(self, msg):
        if not self.origin_set:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.origin_set = True
            self.get_logger().info(
                f"Origin set: {self.origin_lat:.6f}, {self.origin_lon:.6f}")
        self.x, self.y = latlon_to_xy(
            msg.latitude, msg.longitude, self.origin_lat, self.origin_lon)

    # ── Actuator helpers ──────────────────────────────────────────────────────

    def thrust(self, left, right):
        self.pub_l.publish(Float64(data=float(np.clip(left,  MIN_THRUST, MAX_THRUST))))
        self.pub_r.publish(Float64(data=float(np.clip(right, MIN_THRUST, MAX_THRUST))))

    def stop(self):
        self.thrust(0.0, 0.0)
        if self.mode == 'bicycle':
            self.pub_lp.publish(Float64(data=0.0))
            self.pub_rp.publish(Float64(data=0.0))

    # ── Safety ────────────────────────────────────────────────────────────────

    def in_bounds(self):
        d = math.sqrt(self.x**2 + self.y**2)
        if d > BOUNDARY_RADIUS:
            if not self.emergency:
                self.get_logger().warn(f"BOUNDARY exceeded at {d:.1f} m — STOP")
                self.emergency = True
            return False
        if self.emergency and d < BOUNDARY_RADIUS * 0.8:
            self.emergency = False
            self.get_logger().info("Re-entered safe zone")
        return not self.emergency

    # ── Main control loop ─────────────────────────────────────────────────────

    def control_loop(self):
        if not self.origin_set:
            return
        if not self.in_bounds():
            self.stop()
            return
        if self.mode == 'unicycle':
            self._unicycle()
        else:
            self._bicycle()

    # ── Unicycle (lawnmower) controller ───────────────────────────────────────

    def _unicycle(self):
        wp      = self.waypoints[self.wp_index]
        dx      = wp[0] - self.x
        dy      = wp[1] - self.y
        dist    = math.sqrt(dx**2 + dy**2)
        des_hdg = math.atan2(dy, dx)
        hdg_err = wrap(des_hdg - self.heading)

        # ── Waypoint reached ──────────────────────────────────────────────────
        if dist < self.wp_accept:
            self.wp_index  = (self.wp_index + 1) % len(self.waypoints)
            self.uni_state = 'align'
            if self.wp_index == 0:
                self.laps += 1
                self.get_logger().info(f"Lawnmower lap {self.laps} complete!")
            self.get_logger().info(
                f"WP reached → next WP{self.wp_index} | state → ALIGN")
            self.stop()
            return

        # ── ALIGN: in-place spin until heading error < 8 ° ───────────────────
        if self.uni_state == 'align':
            if abs(hdg_err) < math.radians(8.0):
                self.uni_state = 'straight'
                self.get_logger().info("Aligned → STRAIGHT")
            else:
                # Positive hdg_err → target is to the LEFT → spin CCW
                # CCW spin: left thruster backward, right thruster forward
                t = UNI_TURN_THRUST * np.sign(hdg_err)
                self.thrust(-t, t)
                self.get_logger().info(
                    f"[ALIGN] hdg_err={math.degrees(hdg_err):.1f}°",
                    throttle_duration_sec=1.0)

        # ── STRAIGHT: surge + differential heading correction ─────────────────
        elif self.uni_state == 'straight':
            # ── FIX #1 ────────────────────────────────────────────────────────
            # Positive hdg_err → target is to the LEFT → need to turn LEFT
            # → increase LEFT thrust, decrease RIGHT thrust
            # v2 had this inverted (left -= corr, right += corr)
            corr  = float(np.clip(KP_HEADING_UNI * hdg_err * 200.0, -300.0, 300.0))
            left  = UNI_STRAIGHT_THRUST + corr   # ← corrected (was minus)
            right = UNI_STRAIGHT_THRUST - corr   # ← corrected (was plus)
            # ──────────────────────────────────────────────────────────────────
            self.thrust(left, right)
            self.get_logger().info(
                f"[STRAIGHT] dist={dist:.1f}m hdg_err={math.degrees(hdg_err):.1f}°",
                throttle_duration_sec=1.0)
            # Re-align if heading drifts too far
            if abs(hdg_err) > math.radians(20.0):
                self.uni_state = 'align'
                self.get_logger().info("Heading drifted → re-ALIGN")

    # ── Bicycle (circle) controller ───────────────────────────────────────────

    def _bicycle(self):
        wp      = self.waypoints[self.wp_index]
        dx      = wp[0] - self.x
        dy      = wp[1] - self.y
        dist    = math.sqrt(dx**2 + dy**2)
        des_hdg = math.atan2(dy, dx)
        hdg_err = wrap(des_hdg - self.heading)

        if dist < self.wp_accept:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)
            if self.wp_index == 0:
                self.laps += 1
                self.get_logger().info(f"Circle lap {self.laps} complete!")
            self.get_logger().info(f"WP reached → WP{self.wp_index}")

        # Proportional steer angle (radians), clipped to ±0.6 rad
        steer = float(np.clip(KP_HEADING_BIC * hdg_err * 0.3, -0.6, 0.6))
        self.thrust(BIC_CRUISE_THRUST, BIC_CRUISE_THRUST)
        self.pub_lp.publish(Float64(data=steer))
        self.pub_rp.publish(Float64(data=steer))
        self.get_logger().info(
            f"[BICYCLE] dist={dist:.1f}m steer={math.degrees(steer):.1f}°",
            throttle_duration_sec=1.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='unicycle',
                        choices=['unicycle', 'bicycle'])
    args, remaining = parser.parse_known_args()

    # ── FIX #3 ────────────────────────────────────────────────────────────────
    # Guard against double-initialisation (e.g. when run from a launch file
    # that already called rclpy.init()).
    if not rclpy.ok():
        rclpy.init(args=remaining)
    # ──────────────────────────────────────────────────────────────────────────

    node = WaypointController(mode=args.mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
