from isaacsim import SimulationApp

import getpass
# from omni.isaac.kit import SimulationApp


class SimAppManager:
    def __init__(self):
        self.simulation_app = None

    def start_simulation_app(self, livesync_usd: str = None):
        if livesync_usd is None:
            user = getpass.getuser()
            livesync_usd = f"omniverse://localhost/Users/{user}/default_4_0.usd"
        self.simulation_app = SimulationApp({"headless": True, "livesync_usd": livesync_usd})
    
    def stop_simulation_app(self):
        self.simulation_app.close()
        self.simulation_app = None

