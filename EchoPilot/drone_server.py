import asyncio
import httpx
import math
import logging
from mcp.server.fastmcp import FastMCP
from mavsdk import System
from mavsdk.action import ActionError, OrbitYawBehavior

# ==============================================================================
# == Global Objects and Helpers
# ==============================================================================
mcp = FastMCP("PX4DroneControlServer")
drone = System()
is_drone_connected = False
connection_monitor_task = None
initial_connection_task = None
_connection_initialized = False
_connection_string = "udpin://0.0.0.0:14550"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_distance_metres(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1); delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


async def check_connection_health():
    """
    Continuously monitors the drone connection and updates the global flag.
    Attempts to reconnect if connection is lost.
    This task should run continuously and not be cancelled unless shutting down.
    """
    global is_drone_connected, _connection_initialized
    
    reconnect_interval = 5.0  # Check every 5 seconds
    
    # Ensure connection is initialized
    if not _connection_initialized:
        try:
            await drone.connect(system_address=_connection_string)
            _connection_initialized = True
            logger.info("Connection initialized in monitor")
        except Exception as e:
            logger.debug(f"Connection init in monitor: {e}")
    
    while True:
        try:
            await asyncio.sleep(reconnect_interval)
            
            # Check current connection state
            try:
                # Use a timeout to avoid blocking
                try:
                    # Try to get connection state with a short timeout
                    state_iterator = drone.core.connection_state()
                    state = await asyncio.wait_for(state_iterator.__anext__(), timeout=2.0)
                    current_connected = state.is_connected
                    
                    if current_connected != is_drone_connected:
                        is_drone_connected = current_connected
                        if current_connected:
                            logger.info("✅ Drone connection restored!")
                            print("✅ Drone connection restored!")
                        else:
                            logger.warning("⚠️  Drone connection lost!")
                            print("⚠️  Drone connection lost! Attempting to reconnect...")
                except asyncio.TimeoutError:
                    # Timeout doesn't necessarily mean disconnected - could be busy
                    logger.debug("Connection state check timed out (may be normal)")
            except Exception as e:
                logger.debug(f"Connection check error: {e}")
                # Only mark as disconnected if we were previously connected
                # This prevents false negatives during initial connection
                if is_drone_connected:
                    # Try one more verification before marking as disconnected
                    try:
                        # Quick check - if we can't get state, assume disconnected
                        async for state in drone.core.connection_state():
                            if not state.is_connected:
                                is_drone_connected = False
                                logger.warning("⚠️  Connection check failed, marking as disconnected")
                                print("⚠️  Connection lost - will attempt reconnection")
                            break
                    except:
                        # If we can't even check, assume disconnected
                        is_drone_connected = False
                        logger.warning("⚠️  Cannot verify connection, marking as disconnected")
                        print("⚠️  Connection lost - will attempt reconnection")
            
            # Attempt reconnection if disconnected
            if not is_drone_connected:
                try:
                    logger.info("Attempting to reconnect to drone...")
                    # Ensure connection is initialized
                    if not _connection_initialized:
                        await drone.connect(system_address=_connection_string)
                        _connection_initialized = True
                    # Try to reconnect (MAVSDK will handle reconnection automatically)
                    # Just verify the connection state
                    async for state in drone.core.connection_state():
                        if state.is_connected:
                            is_drone_connected = True
                            logger.info("✅ Reconnected to drone!")
                        break
                except Exception as e:
                    logger.debug(f"Reconnection attempt: {e}")
                    
        except asyncio.CancelledError:
            logger.warning("⚠️  Connection monitor cancelled - this should only happen on shutdown")
            # Don't break immediately - try to preserve connection
            # Only break if we're truly shutting down
            break
        except Exception as e:
            logger.error(f"Error in connection monitor: {e}")
            await asyncio.sleep(reconnect_interval)


# ==============================================================================
# == MCP Server Tool Definitions (More Robust and Better Docstrings)
# ==============================================================================

async def check_drone_connection():
    """
    Checks if the drone is actually connected by querying MAVSDK.
    This is more reliable than checking a flag that might be stale.
    Tries multiple methods to verify connection, with proper waiting.
    Also ensures connection is initiated if not already done.
    
    Returns:
        tuple: (is_connected, error_message)
    """
    global is_drone_connected
    
    try:
        # First, ensure we've attempted to connect if not already done
        # Only connect if we haven't connected yet (to avoid multiple connection attempts)
        global _connection_initialized
        if not _connection_initialized:
            try:
                # Try to connect
                await drone.connect(system_address=_connection_string)
                _connection_initialized = True
                logger.debug("Connection initiated in check_drone_connection")
                # Give it a moment to establish
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Connect attempt in check: {e}")
        
        # Method 1: Check connection state from MAVSDK with proper iteration
        # We need to iterate through states until we find a connected one or timeout
        try:
            logger.debug("Checking connection state...")
            connection_timeout = 5.0  # Give it 5 seconds to find connection
            start_time = asyncio.get_event_loop().time()
            
            async for state in drone.core.connection_state():
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > connection_timeout:
                    logger.debug(f"Connection state check timed out after {elapsed:.1f}s")
                    break
                
                if state.is_connected:
                    is_drone_connected = True
                    logger.info("✅ Connection verified via connection state")
                    return True, ""
                else:
                    logger.debug(f"Connection state: not connected (elapsed: {elapsed:.1f}s)")
                    # Continue waiting for connected state
                    
        except asyncio.TimeoutError:
            logger.debug("Connection state iterator timed out")
        except Exception as e:
            logger.debug(f"Connection state check error: {e}, trying telemetry check...")
        
        # Method 2: Try to get telemetry as a more reliable connection test
        # If we can get telemetry, the connection is definitely working
        try:
            logger.debug("Trying telemetry check...")
            # Try to get position with a reasonable timeout
            position = await asyncio.wait_for(
                drone.telemetry.position().__anext__(),
                timeout=3.0
            )
            # If we got here, connection is working
            is_drone_connected = True
            logger.info("✅ Connection verified via telemetry")
            return True, ""
        except asyncio.TimeoutError:
            logger.debug("Telemetry check timed out")
        except StopAsyncIteration:
            logger.debug("Telemetry iterator exhausted")
        except Exception as e:
            logger.debug(f"Telemetry check error: {e}")
        
        # Method 3: Fallback to flag if both checks failed
        # This handles cases where connection is established but state isn't available yet
        if is_drone_connected:
            logger.info("Using cached connection state (flag)")
            return True, ""
        
        # All checks failed - try one more time with a longer wait
        logger.warning("All connection checks failed, attempting one more verification...")
        try:
            # Give it one more chance with a longer timeout
            async for state in drone.core.connection_state():
                if state.is_connected:
                    is_drone_connected = True
                    logger.info("✅ Connection verified on retry")
                    return True, ""
                # Only check once more
                break
        except Exception as e:
            logger.debug(f"Retry check error: {e}")
        
        # All checks failed
        logger.error("❌ Drone connection check failed - all methods exhausted")
        return False, "Drone is not connected. Please ensure PX4 SITL is running and listening on UDP port 14550."
        
    except Exception as e:
        logger.error(f"Unexpected error in connection check: {e}", exc_info=True)
        # Fallback to flag check
        if is_drone_connected:
            logger.info("Using cached connection state after error")
            return True, ""
        return False, f"Drone connection check failed: {str(e)}. Please ensure PX4 SITL is running and listening on UDP port 14550."


@mcp.tool()
async def pre_flight_check() -> dict:
    """
    Performs critical pre-flight safety checks to ensure the drone is armable.
    This verifies GPS lock, home position, and sensor health.
    This should be the first step in almost every mission plan.

    Args: None
    Returns: dict: A JSON object with "status" and "message".
    """
    # Retry connection check up to 3 times with increasing delays
    # This handles cases where connection is still establishing
    max_retries = 3
    connected = False
    error_msg = ""
    
    for attempt in range(max_retries):
        connected, error_msg = await check_drone_connection()
        if connected:
            break
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 1.0  # 1s, 2s delays
            logger.info(f"Connection check failed (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    if not connected:
        logger.error(f"Pre-flight check failed: Drone not connected after {max_retries} attempts")
        return {"status": "Error", "message": error_msg}
    
    logger.info("Starting pre-flight safety checks...")
    print("Performing pre-flight checks...")
    
    try:
        # Verify connection is still active before proceeding
        if not is_drone_connected:
            logger.error("Pre-flight check failed: Connection lost during check")
            return {"status": "Error", "message": "Drone connection lost. Please reconnect."}
        
        # Collect health data with timeout
        health_data = None
        try:
            async for health in drone.telemetry.health():
                health_data = health
                break
        except asyncio.TimeoutError:
            logger.error("Pre-flight check failed: Timeout waiting for health data")
            return {"status": "Error", "message": "Pre-flight check timed out. Connection may be unstable."}
        
        if health_data is None:
            logger.error("Pre-flight check failed: No health data received")
            return {"status": "Error", "message": "Pre-flight check failed: Could not retrieve health data."}
        
        # Detailed component checks
        failed_components = []
        component_status = {}
        
        # Check GPS/Global Position
        if not health_data.is_global_position_ok:
            failed_components.append("GPS/Global Position")
            component_status["gps"] = "FAILED"
            logger.warning("Pre-flight check: GPS/Global Position - FAILED")
            print("  ❌ GPS/Global Position: FAILED")
        else:
            component_status["gps"] = "OK"
            logger.info("Pre-flight check: GPS/Global Position - OK")
            print("  ✅ GPS/Global Position: OK")
        
        # Check Home Position
        if not health_data.is_home_position_ok:
            failed_components.append("Home Position")
            component_status["home_position"] = "FAILED"
            logger.warning("Pre-flight check: Home Position - FAILED")
            print("  ❌ Home Position: FAILED")
        else:
            component_status["home_position"] = "OK"
            logger.info("Pre-flight check: Home Position - OK")
            print("  ✅ Home Position: OK")
        
        # Check if armable
        if not health_data.is_armable:
            failed_components.append("Armable Status")
            component_status["armable"] = "FAILED"
            logger.warning("Pre-flight check: Armable Status - FAILED")
            print("  ❌ Armable Status: FAILED")
        else:
            component_status["armable"] = "OK"
            logger.info("Pre-flight check: Armable Status - OK")
            print("  ✅ Armable Status: OK")
        
        # Check additional health indicators if available
        try:
            # Check if local position is OK (if available)
            if hasattr(health_data, 'is_local_position_ok'):
                if not health_data.is_local_position_ok:
                    failed_components.append("Local Position")
                    component_status["local_position"] = "FAILED"
                    logger.warning("Pre-flight check: Local Position - FAILED")
                    print("  ❌ Local Position: FAILED")
                else:
                    component_status["local_position"] = "OK"
                    logger.info("Pre-flight check: Local Position - OK")
                    print("  ✅ Local Position: OK")
            
            # Check if accelerometer is OK (if available)
            if hasattr(health_data, 'is_accelerometer_calibration_ok'):
                if not health_data.is_accelerometer_calibration_ok:
                    failed_components.append("Accelerometer Calibration")
                    component_status["accelerometer"] = "FAILED"
                    logger.warning("Pre-flight check: Accelerometer Calibration - FAILED")
                    print("  ❌ Accelerometer Calibration: FAILED")
                else:
                    component_status["accelerometer"] = "OK"
                    logger.info("Pre-flight check: Accelerometer Calibration - OK")
                    print("  ✅ Accelerometer Calibration: OK")
            
            # Check if gyroscope is OK (if available)
            if hasattr(health_data, 'is_gyroscope_calibration_ok'):
                if not health_data.is_gyroscope_calibration_ok:
                    failed_components.append("Gyroscope Calibration")
                    component_status["gyroscope"] = "FAILED"
                    logger.warning("Pre-flight check: Gyroscope Calibration - FAILED")
                    print("  ❌ Gyroscope Calibration: FAILED")
                else:
                    component_status["gyroscope"] = "OK"
                    logger.info("Pre-flight check: Gyroscope Calibration - OK")
                    print("  ✅ Gyroscope Calibration: OK")
            
            # Check if magnetometer is OK (if available)
            if hasattr(health_data, 'is_magnetometer_calibration_ok'):
                if not health_data.is_magnetometer_calibration_ok:
                    failed_components.append("Magnetometer Calibration")
                    component_status["magnetometer"] = "FAILED"
                    logger.warning("Pre-flight check: Magnetometer Calibration - FAILED")
                    print("  ❌ Magnetometer Calibration: FAILED")
                else:
                    component_status["magnetometer"] = "OK"
                    logger.info("Pre-flight check: Magnetometer Calibration - OK")
                    print("  ✅ Magnetometer Calibration: OK")
        except AttributeError:
            # Some health fields may not be available in all MAVSDK versions
            logger.debug("Some health fields not available in this MAVSDK version")
        
        # Final assessment
        if failed_components:
            error_message = (
                f"Pre-flight checks failed. Failed components: {', '.join(failed_components)}. "
                f"Please check sensors/calibration. Component status: {component_status}"
            )
            logger.error(f"Pre-flight check failed. Components: {failed_components}")
            print(f"-- Pre-flight check FAILED. Failed components: {', '.join(failed_components)}")
            return {"status": "Error", "message": error_message}
        else:
            success_message = "All pre-flight checks passed. Drone is armable."
            logger.info("Pre-flight check: All checks passed - Drone is armable")
            print("-- ✅ All pre-flight checks passed. Drone is armable.")
            return {"status": "Success", "message": success_message}
            
    except asyncio.TimeoutError:
        logger.error("Pre-flight check failed: Timeout exception")
        return {"status": "Error", "message": "Pre-flight check timed out. Connection may be unstable."}
    except Exception as e:
        logger.error(f"Pre-flight check error: {e}", exc_info=True)
        # Don't mark as disconnected on error - let the monitor handle it
        return {"status": "Error", "message": f"Pre-flight check failed: {str(e)}"}

@mcp.tool()
async def arm_and_takeoff(altitude_meters: float) -> dict:
    """
    Arms the drone's motors and takes off vertically to a specific altitude.
    Waits until the drone reaches the target altitude before completing.

    Args:
        altitude_meters (float): The target altitude in meters relative to the ground.

    Returns: dict: A JSON object with "status" confirming successful takeoff.
    """

    connected, error_msg = await check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}
    try:
        print("-- Arming drone...")
        await drone.action.arm()
        await asyncio.sleep(1)
        print(f"-- Taking off to {altitude_meters} meters...")
        await drone.action.set_takeoff_altitude(altitude_meters)
        await drone.action.takeoff()
        print(f"-- Monitoring altitude... Target: {altitude_meters}m")
        async for position in drone.telemetry.position():
            if position.relative_altitude_m >= altitude_meters * 0.95:
                print(f"-- Target altitude of {altitude_meters}m reached!")
                break
        return {"status": "Success", "message": "Arm and takeoff successful."}
    except ActionError as e: return {"status": "Error", "message": f"Arm/Takeoff failed: {e}"}


@mcp.tool()
async def get_coordinates_for_location(location_name: str) -> dict:
    """
    Converts a human-readable location name (e.g., "Eiffel Tower") into GPS coordinates.

    Args:
        location_name (str): The name of the location (e.g., "Eiffel Tower, Paris").

    Returns: dict: A JSON object with "status", "latitude", "longitude", and "address".
    """

    url = f"https://nominatim.openstreetmap.org/search?q={location_name.replace(' ', '+')}&format=json&limit=1"
    headers = {'User-Agent': 'DroneControlMCP/1.0'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            if data: return {"status": "Success", "latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"]), "address": data[0]["display_name"]}
            return {"status": "Error", "message": f"Could not find coordinates for '{location_name}'."}
    except Exception as e: return {"status": "Error", "message": f"Geocoding or network request failed: {e}"}


@mcp.tool()
async def fly_to_coordinates(latitude: float, longitude: float, altitude_meters: float | None = None, velocity_ms: float | None = None) -> dict:
    """
    Flies the drone to a specific GPS coordinate. Waits for arrival before completing.

    Args:
        latitude (float): The target latitude.
        longitude (float): The target longitude.
        altitude_meters (float, optional): The target absolute altitude (AMSL). If not provided, maintains current altitude.
        velocity_ms (float, optional): The speed for this flight leg in m/s. If not provided, a default speed of 5.0 m/s will be used.

    Returns: dict: A JSON object with "status" confirming successful arrival.
    """
    connected, error_msg = check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}


    speed_to_use = velocity_ms if velocity_ms is not None else 5.0

    final_altitude = altitude_meters
    if final_altitude is None:
        try:
            position = await drone.telemetry.position().__anext__()
            final_altitude = position.absolute_altitude_m
        except StopAsyncIteration: return {"status": "Error", "message": "Failed to get current altitude."}

    print(f"-- Flying to {latitude}, {longitude} at {speed_to_use} m/s...")
    try:
        # Set speed for this specific flight leg
        await drone.action.set_current_speed(speed_to_use)
        await drone.action.goto_location(latitude, longitude, final_altitude, 0)

        arrival_threshold_meters = 5.0
        while True:
            await asyncio.sleep(2)
            try:
                # Heartbeat ping
                print("-- Sending heartbeat ping to drone...")
                await drone.action.set_current_speed(speed_to_use)

                current_pos = await drone.telemetry.position().__anext__()
                distance_to_target = get_distance_metres(current_pos.latitude_deg, current_pos.longitude_deg, latitude, longitude)
                print(f"-- Distance to target: {distance_to_target:.2f} meters...")

                if distance_to_target < arrival_threshold_meters:
                    print("-- Arrived at target location!")
                    break
            except StopAsyncIteration:
                return {"status": "Error", "message": "Telemetry lost during flight."}

        print("-- Arrived. Stabilizing for 2 seconds...")
        await asyncio.sleep(2)
        return {"status": "Success", "message": "Navigation successful and arrival confirmed."}
    except ActionError as e:
        return {"status": "Error", "message": f"Goto location failed: {e}"}


@mcp.tool()
async def fly_relative(forward_meters: float = 0, right_meters: float = 0, down_meters: float = 0) -> dict:
    """
    Commands the drone to fly a certain distance relative to its current position and heading.

    Args:
        forward_meters (float, optional): Distance to fly forward in meters. Use a negative value to fly backward. Defaults to 0.
        right_meters (float, optional): Distance to fly to the right in meters. Use a negative value to fly left. Defaults to 0.
        down_meters (float, optional): Distance to fly down in meters. Use a negative value to fly up. Defaults to 0.

    Returns:
        dict: A JSON object confirming the relative move is complete.
    """
    connected, error_msg = check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}

    print(f"-- Flying relative: {forward_meters}m forward, {right_meters}m right, {down_meters}m down...")

    try:
        # Get the current position and heading
        position = await drone.telemetry.position().__anext__()
        heading_deg = await drone.telemetry.heading().__anext__()

        # Simple trigonometry to calculate the new GPS coordinate
        earth_radius = 6378137.0
        # Calculate offset in radians
        lat_offset = (forward_meters * math.cos(math.radians(heading_deg.heading_deg)) - right_meters * math.sin(math.radians(heading_deg.heading_deg))) / earth_radius
        lon_offset = (forward_meters * math.sin(math.radians(heading_deg.heading_deg)) + right_meters * math.cos(math.radians(heading_deg.heading_deg))) / (earth_radius * math.cos(math.radians(position.latitude_deg)))

        # Convert radians to degrees and add to current position
        new_latitude = position.latitude_deg + math.degrees(lat_offset)
        new_longitude = position.longitude_deg + math.degrees(lon_offset)

        # Adjust altitude
        new_altitude = position.absolute_altitude_m - down_meters

        print(f"-- Calculated new target: Lat {new_latitude}, Lon {new_longitude}")

        # Reuse the robust goto_location logic to fly to the new point
        # You can call other async functions directly
        return await fly_to_coordinates(new_latitude, new_longitude, new_altitude)

    except (ActionError, StopAsyncIteration) as e:
        return {"status": "Error", "message": f"Failed to execute relative flight: {e}"}

@mcp.tool()
async def do_orbit(latitude: float, longitude: float, radius_meters: float, velocity_ms: float | None = None) -> dict:
    """
    Flies a circle around a GPS point for a fixed duration of 30 seconds.

    Args:
        latitude (float): The latitude of the center point.
        longitude (float): The longitude of the center point.
        radius_meters (float): The radius of the circle in meters.
        velocity_ms (float, optional): The speed for the orbit in m/s. If not provided, a default of 5.0 m/s is used.

    Returns:
        dict: A JSON object confirming the orbit action was executed for 30 seconds.
    """
    connected, error_msg = check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}

    speed_to_use = velocity_ms if velocity_ms is not None else 5.0

    print(f"-- Initiating orbit at {speed_to_use} m/s...")
    try:
        position = await drone.telemetry.position().__anext__()
        absolute_altitude_m = position.absolute_altitude_m

        # Start the orbit action
        await drone.action.do_orbit(
            radius_m=radius_meters,
            velocity_ms=speed_to_use,
            yaw_behavior=OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER,
            latitude_deg=latitude,
            longitude_deg=longitude,
            absolute_altitude_m=absolute_altitude_m
        )


        orbit_duration = 60  # You can change this to any duration you want
        print(f"-- Orbiting for a fixed duration of {orbit_duration} seconds.")
        await asyncio.sleep(orbit_duration)

        print("-- Orbit time complete. Holding position to stabilize...")
        await drone.action.hold()
        await asyncio.sleep(2)

        return {"status": "Success", "message": f"Orbit action completed after {orbit_duration} seconds."}
    except (ActionError, StopAsyncIteration) as e:
        return {"status": "Error", "message": f"Orbit failed: {e}"}

@mcp.tool()
async def land() -> dict:
    """
    Commands the drone to land at its current position.

    Args: None
    Returns: dict: A JSON object with "status" confirming a successful landing.
    """

    connected, error_msg = check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}
    print("-- Landing command issued...")
    try:
        await drone.action.land()
        print("-- Monitoring for landing completion...")
        async for is_armed in drone.telemetry.armed():
            if not is_armed:
                print("-- Landing and disarm confirmed!")
                break
        return {"status": "Success", "message": "Landing successful."}
    except ActionError as e: return {"status": "Error", "message": f"Landing failed: {e}"}

@mcp.tool()
async def return_to_launch() -> dict:
    """
    Commands the drone to return to its original take-off location and land.

    Args: None
    Returns: dict: A JSON object confirming a successful return and landing.
    """

    connected, error_msg = await check_drone_connection()
    if not connected:
        return {"status": "Error", "message": error_msg}
    print("-- Return to Launch (RTL) command issued...")
    try:
        await drone.action.return_to_launch()
        print("-- Monitoring for landing completion at launch point...")
        async for is_armed in drone.telemetry.armed():
            if not is_armed:
                print("-- RTL landing and disarm confirmed!")
                break
        return {"status": "Success", "message": "Return to launch successful."}
    except ActionError as e: return {"status": "Error", "message": f"RTL failed: {e}"}

async def initial_drone_connection():
    """
    Non-blocking initial drone connection attempt.
    Runs in background while MCP server starts.
    This should only run once and not be cancelled.
    """
    global is_drone_connected, _connection_initialized
    
    print(f"Attempting to connect to drone at {_connection_string}...")
    
    try:
        # Connect to the drone (only if not already initialized)
        if not _connection_initialized:
            await drone.connect(system_address=_connection_string)
            _connection_initialized = True
            logger.info("Initial connection established")
        
        # Wait for connection with timeout
        connection_timeout = 10.0  # 10 seconds timeout
        start_time = asyncio.get_event_loop().time()
        
        print("Waiting for drone connection...")
        async for state in drone.core.connection_state():
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > connection_timeout:
                print(f"⚠️  Connection timeout after {connection_timeout} seconds.")
                print("   Make sure PX4 SITL is running and listening on UDP port 14550.")
                print("   Connection monitoring will continue in the background.")
                break
            
            if state.is_connected:
                print("✅ Drone Connected!")
                is_drone_connected = True
                break
        
        if not is_drone_connected:
            print("⚠️  Warning: Could not connect to drone initially.")
            print("   The MCP server will start anyway, and connection monitoring will continue.")
            print("   To connect to drone:")
            print("   1. Start PX4 SITL: make px4_sitl gazebo")
            print("   2. Ensure it's listening on UDP port 14550")
            print("   3. The server will automatically reconnect when available")
    except asyncio.CancelledError:
        logger.warning("Initial connection task cancelled - connection may still be active")
        # Don't reset flags on cancellation - connection might still be good
    except Exception as e:
        print(f"❌ Error during initial drone connection: {e}")
        print("   Connection monitoring will continue in the background.")
        # Don't set is_drone_connected to False here - let the monitor handle it


async def main():
    """
    Main entry point for the MCP server.
    Connects to the drone over UDP on port 14550 and starts the MCP server.
    Maintains connection monitoring in the background.
    This function should run continuously and not restart.
    """
    global is_drone_connected, connection_monitor_task, initial_connection_task
    
    # Start initial connection attempt - give it a short time to connect quickly
    # This helps if the drone is already running
    # Only create task if it doesn't already exist
    if initial_connection_task is None or initial_connection_task.done():
        initial_connection_task = asyncio.create_task(initial_drone_connection())
    
    # Wait a short time (2 seconds) to see if connection establishes quickly
    # This doesn't block the server startup, but gives connection a head start
    try:
        await asyncio.wait_for(asyncio.sleep(2.0), timeout=2.0)
    except:
        pass
    
    # Start connection monitor in the background
    # This will continuously check and maintain the connection
    # Only create task if it doesn't already exist or is done
    if connection_monitor_task is None or connection_monitor_task.done():
        connection_monitor_task = asyncio.create_task(check_connection_health())
        print("✅ Connection monitor started (will maintain connection automatically)")
    else:
        print("✅ Connection monitor already running")
    
    # Start MCP server immediately (non-blocking)
    # This allows the agent to test and use other tools (like geocoding) even if drone isn't connected
    # The server should run continuously and not restart
    print("\n" + "="*60)
    print("Starting MCP server. Awaiting commands from the agent...")
    print("="*60 + "\n")
    
    try:
        # Run the MCP server - this should block and run continuously
        await mcp.run_stdio_async()
    except KeyboardInterrupt:
        print("\n⚠️  MCP server interrupted by user")
        raise
    except Exception as e:
        print(f"❌ MCP server error: {e}")
        logger.error(f"MCP server error: {e}", exc_info=True)
        raise
    finally:
        # Only clean up on actual shutdown (not on restart)
        logger.info("MCP server shutting down - cleaning up tasks...")
        if initial_connection_task and not initial_connection_task.done():
            initial_connection_task.cancel()
            try:
                await initial_connection_task
            except (asyncio.CancelledError, Exception):
                pass
        if connection_monitor_task and not connection_monitor_task.done():
            connection_monitor_task.cancel()
            try:
                await connection_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Cleanup complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer terminated by user.")
