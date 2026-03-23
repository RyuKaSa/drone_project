"""
ESP Drone - Live Telemetry Monitor
===================================
Configurable Hz and variable selection.
Requires: espdrone-lib-python (edlib)

Configure the SETTINGS section below before running.
"""

import time
import sys
import edlib.crtp
from edlib.espdrone import Espdrone
from edlib.espdrone.syncEspdrone import SyncEspdrone
from edlib.espdrone.log import LogConfig

# ============================================================
#  SETTINGS - Edit these to configure your telemetry stream
# ============================================================

DRONE_IP = "192.168.43.42"

# Update rate in Hz
UPDATE_HZ = 100

# Which variables to stream. Comment/uncomment lines to select.
# Each group becomes a separate log config (max 26 bytes per config).
# Available types: 'float', 'uint32_t', 'int16_t', 'uint16_t', 'uint8_t', 'int8_t'

LOG_GROUPS = {
    # --- IMU Raw ---
    "imu": [
        # ('gyro.x', 'float'),
        # ('gyro.y', 'float'),
        ('gyro.z', 'float'),
        # ('acc.x', 'float'),
        # ('acc.y', 'float'),
        ('acc.z', 'float'),
    ],

    # --- Attitude Estimates ---
    "attitude": [
        ('stabilizer.roll', 'float'),
        ('stabilizer.pitch', 'float'),
        ('stabilizer.yaw', 'float'),
        # ('stabilizer.thrust', 'float'),
    ],

    # --- Motors ---
    # "motors": [
    #     ('motor.m1', 'uint32_t'),
    #     ('motor.m2', 'uint32_t'),
    #     ('motor.m3', 'uint32_t'),
    #     ('motor.m4', 'uint32_t'),
    # ],

    # --- Battery ---
    # "battery": [
    #     ('pm.vbat', 'float'),
    #     ('pm.batteryLevel', 'uint8_t'),
    #     ('pm.state', 'int8_t'),
    # ],

    # --- State Estimate (Kalman) ---
    "state": [
        # ('stateEstimate.x', 'float'),
        # ('stateEstimate.y', 'float'),
        # ('stateEstimate.z', 'float'),
        # ('stateEstimate.vx', 'float'),
        # ('stateEstimate.vy', 'float'),
        ('stateEstimate.vz', 'float'),
    ],

    # --- Barometer --- doesnt work, all 0
    # "baro": [
    #     ('baro.pressure', 'float'),
    #     ('baro.temp', 'float'),
    #     ('baro.asl', 'float'),
    # ],

    # --- Controller internals ---
    # "controller": [
    #     ('controller.cmd_roll', 'float'),
    #     ('controller.cmd_pitch', 'float'),
    #     ('controller.cmd_yaw', 'float'),
    #     ('controller.cmd_thrust', 'float'),
    # ],

    # --- System health ---
    # "system": [
    #     ('sys.canfly', 'int8_t'),
    #     ('sys.armed', 'int8_t'),
    #     ('stabilizer.intToOut', 'uint32_t'),
    #     ('crtp.rxRate', 'uint16_t'),
    #     ('crtp.txRate', 'uint16_t'),
    # ],
}

# Display format: 'table' for aligned columns, 'raw' for simple print
DISPLAY_FORMAT = 'table'

# ============================================================
#  END SETTINGS
# ============================================================


def startup_drone(sed):
    """
    Run prop test to set sys.canfly=1 and unfreeze the stabilizer.
    Must be called once per power cycle.
    """
    print("Running startup prop test...")
    print("  (motors will briefly spin in sequence)")

    # Trigger the prop test
    sed.ed.param.set_value('health.startPropTest', 1)

    # Wait for all 4 motors to test
    time.sleep(5)

    # Verify canfly
    log_check = LogConfig(name='Check', period_in_ms=100)
    log_check.add_variable('sys.canfly', 'int8_t')

    result = [None]

    def check_cb(timestamp, data, logconf):
        result[0] = data['sys.canfly']

    sed.ed.log.add_config(log_check)
    log_check.data_received_cb.add_callback(check_cb)
    log_check.start()

    # Send keep-alive while checking
    for _ in range(10):
        
        time.sleep(0.1)

    log_check.stop()

    if result[0] == 1:
        print("  Prop test PASSED. sys.canfly=1. Stabilizer is running.\n")
        return True
    else:
        print(f"  WARNING: sys.canfly={result[0]}. Stabilizer may not be running.")
        print("  Check motor connections and try power cycling the drone.\n")
        return False


def main():
    period_ms = max(10, int(1000 / UPDATE_HZ))
    actual_hz = 1000 / period_ms
    print(f"ESP Drone Telemetry Monitor")
    print(f"  Target: {DRONE_IP}")
    print(f"  Requested: {UPDATE_HZ} Hz -> period: {period_ms} ms -> actual: {actual_hz:.1f} Hz")
    print(f"  Groups: {', '.join(LOG_GROUPS.keys())}")
    print()

    edlib.crtp.init_drivers()

    with SyncEspdrone(DRONE_IP, ed=Espdrone(rw_cache='./cache')) as sed:
        print("Connected!\n")

        # --- Startup sequence ---
        startup_drone(sed)

        # --- Build log configs ---
        latest_data = {}
        configs = []
        total_vars = 0

        for group_name, variables in LOG_GROUPS.items():
            log_conf = LogConfig(name=group_name, period_in_ms=period_ms)
            for var_name, var_type in variables:
                log_conf.add_variable(var_name, var_type)
                total_vars += 1

            def make_callback(gname, varlist):
                def cb(timestamp, data, logconf):
                    latest_data['_timestamp'] = timestamp
                    for var_name, _ in varlist:
                        latest_data[var_name] = data[var_name]
                return cb

            sed.ed.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(make_callback(group_name, variables))
            log_conf.start()
            configs.append(log_conf)
            print(f"  Started logging: {group_name} ({len(variables)} vars)")

        print(f"\n  Total: {total_vars} variables at {actual_hz:.0f} Hz")
        print(f"\n{'='*70}")
        print("STREAMING LIVE — tilt/move the drone to see changes (Ctrl+C to stop)")
        print(f"{'='*70}\n")

        # --- Collect all variable names for display ---
        all_vars = []
        for group_name, variables in LOG_GROUPS.items():
            for var_name, _ in variables:
                all_vars.append(var_name)

        # --- Print header ---
        if DISPLAY_FORMAT == 'table':
            header = "  timestamp  | " + " | ".join(
                f"{v.split('.')[-1]:>8}" for v in all_vars
            )
            print(header)
            print("-" * len(header))

        # --- Main loop ---
        sample_count = 0
        t_start = time.time()

        try:
            while True:
                

                if latest_data and '_timestamp' in latest_data:
                    ts = latest_data['_timestamp']

                    if DISPLAY_FORMAT == 'table':
                        vals = []
                        for v in all_vars:
                            val = latest_data.get(v, None)
                            if val is None:
                                vals.append(f"{'---':>8}")
                            elif isinstance(val, float):
                                vals.append(f"{val:8.3f}")
                            else:
                                vals.append(f"{val:>8}")
                        print(f"  {ts:>9}  | " + " | ".join(vals))
                    else:
                        parts = []
                        for v in all_vars:
                            val = latest_data.get(v, '---')
                            short = v.split('.')[-1]
                            if isinstance(val, float):
                                parts.append(f"{short}={val:.3f}")
                            else:
                                parts.append(f"{short}={val}")
                        print(f"[{ts}] {' | '.join(parts)}")

                    sample_count += 1

                # Sleep to roughly match the update rate for printing
                # (data arrives via callbacks independently)
                time.sleep(period_ms / 1000.0)

        except KeyboardInterrupt:
            elapsed = time.time() - t_start
            effective_hz = sample_count / elapsed if elapsed > 0 else 0
            print(f"\n\nStopped. {sample_count} samples in {elapsed:.1f}s = {effective_hz:.1f} Hz effective")

        # --- Cleanup ---
        for conf in configs:
            conf.stop()
        

    print("Disconnected. Done.")


if __name__ == '__main__':
    main()