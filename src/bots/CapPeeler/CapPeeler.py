from robocode_tank_royale.bot_api import Color
from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import ScannedBotEvent, HitBotEvent, WonRoundEvent, RoundEndedEvent, DeathEvent


# ------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------
# A sample robot originally made for Robocode by Mathew Nelson,
# ported to Robocode Tank Royale.
#
# Locks onto a robot, moves close, fires when close.
# ------------------------------------------------------------------
class CapPeeler(Bot):
    def __init__(self) -> None:
        super().__init__()
        self._count: int = 0          # How long we've been searching for our target
        self._gun_turn_amt: float = 10.0  # How much to turn our gun when searching
        self._track_id: int | None = None  # ID of the robot we're currently tracking
        # Store scanned bot data to process in main loop
        self._scanned_target_x: float | None = None
        self._scanned_target_y: float | None = None
        self._scanned_target_distance: float | None = None
        self._scanned_target_gun_bearing: float | None = None
        self._scanned_target_bearing: float | None = None
        # Store hit bot data to process in main loop
        self._hit_bot_x: float | None = None
        self._hit_bot_y: float | None = None
        self._hit_bot_id: int | None = None

    def run(self) -> None:
        """Tracker's main run function."""
        # Set colors
        red = Color.from_rgb(0xFF, 0x00, 0x00)
        white = Color.from_rgb(0xFF, 0xFF, 0xFF)
        self.body_color = red
        self.turret_color = white
        self.radar_color = Color.from_rgb(200, 200, 70)
        self.scan_color = Color.from_rgb(255, 255, 255)
        self.bullet_color = Color.from_rgb(0, 0, 255)

        # Prepare gun
        self._track_id = None  # Initialize to not tracking anyone
        self._gun_turn_amt = 10.0  # Initialize gun turn to 10
        self._count = 0

        # Loop forever
        while self.running:
            # Process hit bot event data if available
            if self._hit_bot_x is not None and self._hit_bot_y is not None:
                # Only print if it's not already our target
                if self._track_id is not None and self._track_id != self._hit_bot_id:
                    print(f"Tracking {self._hit_bot_id} due to collision")

                # Set the target
                self._track_id = self._hit_bot_id

                # Back up a bit
                gun_bearing = self.gun_bearing_to(self._hit_bot_x, self._hit_bot_y)
                self._gun_turn_amt = -gun_bearing
                self.turn_gun_right(-gun_bearing)
                self.fire(3)
                self.back(50)

                # Clear hit bot data
                self._hit_bot_x = None
                self._hit_bot_y = None
                self._hit_bot_id = None

            # Process scanned bot event data if available
            elif self._scanned_target_x is not None:
                distance = self._scanned_target_distance
                gun_bearing = self._scanned_target_gun_bearing
                bearing = self._scanned_target_bearing

                # If our target is too far away, turn and move toward it
                if distance > 150:
                    self._gun_turn_amt = -gun_bearing
                    self.turn_gun_right(-gun_bearing)
                    self.turn_left(bearing)
                    self.forward(distance - 140)
                else:
                    # Our target is close
                    self._gun_turn_amt = -gun_bearing
                    self.turn_gun_right(-gun_bearing)
                    self.fire(3)

                    # Our target is too close! Back up
                    if distance < 100:
                        if bearing >= -90 and bearing < 90:
                            self.back(40)
                        else:
                            self.forward(40)

                # Clear scanned target data
                self._scanned_target_x = None
                self._scanned_target_y = None
                self._scanned_target_distance = None
                self._scanned_target_gun_bearing = None
                self._scanned_target_bearing = None

            else:
                # Default scanning behavior when no target data
                # Turn the gun (looks for enemy)
                self.turn_gun_right(self._gun_turn_amt)
                # Keep track of how long we've been looking
                self._count += 1
                # If we haven't seen our target for 2 turns, look left
                if self._count > 2:
                    self._gun_turn_amt = -10
                # If we still haven't seen our target for 5 turns, look right
                if self._count > 5:
                    self._gun_turn_amt = 10
                # If we *still* haven't seen our target after 10 turns, find another target
                if self._count > 11:
                    self._track_id = None

    def on_scanned_bot(self, e: ScannedBotEvent) -> None:
        """Here's the good stuff - store scanned bot data for main loop to process."""
        # If we have a target, and this isn't it, return immediately
        # so we can get more ScannedBotEvents.
        if self._track_id is not None and e.scanned_bot_id != self._track_id:
            return

        # If we don't have a target, well, now we do!
        if self._track_id is None:
            self._track_id = e.scanned_bot_id
            print(f"🔍Tracking {self._track_id}")

        # This is our target. Reset count (see the run method).
        self._count = 0

        # Store the scanned bot data for the main loop to process
        # DO NOT call blocking actions here - that causes event queue errors!
        self._scanned_target_x = e.x
        self._scanned_target_y = e.y
        self._scanned_target_distance = self.distance_to(e.x, e.y)
        self._scanned_target_gun_bearing = self.gun_bearing_to(e.x, e.y)
        self._scanned_target_bearing = self.bearing_to(e.x, e.y)


    def on_hit_bot(self, e: HitBotEvent) -> None:
        """Set the bot we hit as our new target - store data for main loop to process."""
        # Store hit bot data for the main loop to process
        # DO NOT call blocking actions here - that causes event queue errors!
        self._hit_bot_x = e.x
        self._hit_bot_y = e.y
        self._hit_bot_id = e.victim_id

    def on_death(self, e: DeathEvent) -> None:
        """We died -> print a message."""
        print(f"🔥Died on turn {e.turn_number}")
        del e

    def on_won_round(self, e: WonRoundEvent) -> None:
        """Do a victory dance."""
        print(f"🏆Won on turn {e.turn_number}")
        for _ in range(50):
            self.turn_right(30)
            self.turn_left(30)
        del e

    def on_round_ended(self, e: RoundEndedEvent) -> None:
        """ Round ended """
        print(f"❌Round {e.round_number} ended")
        del e

def main() -> None:
    bot = Tracker()
    bot.start()


if __name__ == "__main__":
    main()
