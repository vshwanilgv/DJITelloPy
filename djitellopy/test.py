from djitellopy import Tello


drone = Tello()

drone.connect()
print(f"Battery:")

def get_battery(self) -> int:
        """Get current battery percentage
        Returns:
            int: 0-100
        """
        return self.get_state_field('bat')
print("Battery check...")
battery_percentage = drone.get_battery()
print(f"Battery: {battery_percentage}%")