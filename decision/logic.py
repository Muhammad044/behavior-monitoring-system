import time

class DecisionMaker:
    def __init__(self):
        self.fall_start_time = None
        self.alert_sent = False

    def process(self, name, action):
        # If the person is fallen, start a timer
        if action == "FALLEN":
            if self.fall_start_time is None:
                self.fall_start_time = time.time()
            
            # Check if they have been down for more than 3 seconds
            elapsed = time.time() - self.fall_start_time
            if elapsed > 3.0 and not self.alert_sent:
                self.alert_sent = True
                return f"CRITICAL ALERT! {name} has been down for {int(elapsed)}s!"
        else:
            # Reset if they stand back up
            self.fall_start_time = None
            self.alert_sent = False
        
        return None