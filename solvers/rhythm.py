import logging
import random
from classes import Solver, State

logger = logging.getLogger(__name__)


class RhythmState(State):
    def __init__(self, data: dict):
        # Rhythm classification (0=normal sinus, 1=afib, 2=vfib, etc.)
        self._rhythm_type = data.get("rhythm_type", 0)
        # PR interval in milliseconds (normal 120-200)
        self._pr_interval = data.get("pr_interval", 160.0)
        # QRS duration in milliseconds (normal 60-100)
        self._qrs_duration = data.get("qrs_duration", 80.0)
        # QT interval in milliseconds (normal ~400)
        self._qt_interval = data.get("qt_interval", 400.0)
        # Heart block degree (0=none, 1=first degree, etc.)
        self._heart_block = data.get("heart_block", 0)
        # R-R variability for rhythm irregularity
        self._rr_variability = data.get("rr_variability", 0.0)

    @property
    def state(self) -> dict:
        return {
            "rhythm_type": self._rhythm_type,
            "pr_interval": self._pr_interval,
            "qrs_duration": self._qrs_duration,
            "qt_interval": self._qt_interval,
            "heart_block": self._heart_block,
            "rr_variability": self._rr_variability
        }


class RhythmSolver(Solver):
    def __init__(self,
                 arrhythmia_threshold: float = 0.7,
                 conduction_recovery_rate: float = 0.05):
        """
        Initialize rhythm solver with conduction parameters
        
        :param arrhythmia_threshold: Threshold for developing arrhythmias
        :param conduction_recovery_rate: Rate of conduction recovery
        """
        self._state = RhythmState({})
        self.arrhythmia_threshold = arrhythmia_threshold
        self.conduction_recovery_rate = conduction_recovery_rate
        
        # Constants for rhythm types
        self.SINUS = 0
        self.AFIB = 1
        self.VFIB = 2
        self.VTACH = 3
        self.ARREST = 4

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        rs = RhythmState(state)
        
        # Get relevant physiological parameters if available
        heart_rate = state.get("heart_rate", 80.0)
        potassium = state.get("potassium", 4.0)
        oxygen_debt = state.get("oxygen_debt", 0.0)
        
        # Calculate probability of rhythm disturbance
        # Factors: high/low HR, K+ abnormalities, oxygen debt
        arrhythmia_risk = 0.0
        if heart_rate > 150 or heart_rate < 40:
            arrhythmia_risk += 0.3
        if potassium > 6.0 or potassium < 2.5:
            arrhythmia_risk += 0.4
        arrhythmia_risk += min(0.5, oxygen_debt / 100.0)
        
        # Determine new rhythm
        new_rhythm = rs.state["rhythm_type"]
        if random.random() < arrhythmia_risk * dt:
            if arrhythmia_risk > self.arrhythmia_threshold:
                if potassium > 7.0:  # Severe hyperkalemia
                    new_rhythm = self.ARREST
                elif oxygen_debt > 50:  # Severe hypoxia
                    new_rhythm = self.VFIB
                elif heart_rate > 180:  # Extreme tachycardia
                    new_rhythm = self.VTACH
                else:
                    new_rhythm = self.AFIB
        elif new_rhythm != self.SINUS:
            # Chance of spontaneous conversion to sinus
            if random.random() < self.conduction_recovery_rate * dt:
                new_rhythm = self.SINUS
        
        # Update conduction intervals based on rhythm and conditions
        new_pr = rs.state["pr_interval"]
        new_qrs = rs.state["qrs_duration"]
        new_qt = rs.state["qt_interval"]
        new_block = rs.state["heart_block"]
        new_variability = rs.state["rr_variability"]
        
        if new_rhythm == self.SINUS:
            # PR interval affected by conduction and K+
            pr_change = ((160.0 - new_pr) * 
                        self.conduction_recovery_rate * dt)
            new_pr += pr_change
            
            # QRS widens with conduction disease
            qrs_change = ((80.0 - new_qrs) * 
                         self.conduction_recovery_rate * dt)
            new_qrs += qrs_change
            
            # QT affected by heart rate and K+
            qt_target = 400.0 - (0.5 * (heart_rate - 60))
            if potassium < 3.5:  # Hypokalemia prolongs QT
                qt_target += (3.5 - potassium) * 50
            qt_change = (qt_target - new_qt) * 0.1 * dt
            new_qt += qt_change
            
            new_variability = max(0, new_variability - 0.2 * dt)
            
        elif new_rhythm == self.AFIB:
            new_variability = min(1.0, new_variability + 0.3 * dt)
            new_pr = 0  # No P waves in AFib
            
        elif new_rhythm == self.VFIB:
            new_variability = 1.0
            new_pr = 0
            new_qrs = 300  # Chaotic, wide complex
            
        elif new_rhythm == self.VTACH:
            new_variability = 0.1
            new_pr = 0
            new_qrs = min(200, new_qrs + 20 * dt)
            
        elif new_rhythm == self.ARREST:
            new_variability = 0
            new_pr = 0
            new_qrs = 0
            new_qt = 0

        # Heart block progression
        if potassium > 6.0:  # Hyperkalemia can cause heart block
            if random.random() < 0.1 * dt:
                new_block = min(3, new_block + 1)
        elif new_block > 0:  # Possible recovery
            if random.random() < self.conduction_recovery_rate * dt:
                new_block = max(0, new_block - 1)

        logger.debug(
            f"RhythmSolver: type={new_rhythm}, "
            f"PR={new_pr:.0f}, QRS={new_qrs:.0f}, "
            f"QT={new_qt:.0f}, block={new_block}"
        )

        return RhythmState({
            "rhythm_type": new_rhythm,
            "pr_interval": new_pr,
            "qrs_duration": new_qrs,
            "qt_interval": new_qt,
            "heart_block": new_block,
            "rr_variability": new_variability
        })