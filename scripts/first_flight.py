import asyncio
from mavsdk import System


async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    print("Waiting for global position lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break

    print("Arming.")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(15)

    print("Transitioning to fixed-wing.")
    await drone.action.transition_to_fixedwing()
    await asyncio.sleep(20)

    print("Transitioning back to multicopter.")
    await drone.action.transition_to_multicopter()
    await asyncio.sleep(10)

    await drone.action.land()


if __name__ == "__main__":
    asyncio.run(main())
