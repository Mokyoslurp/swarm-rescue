class FilterParameters:
    """A class that holds all necessary parameters for the filters to work"""

    def __init__(
        self,
        mass: float = 50.0,
        speed_damping: float = 0.05,
        force_gain: float = 30,
        angular_speed_gain: float = 0.3 * 0.6,
        time_step: float = 1,
    ) -> None:
        self.mass = mass
        self.inv_mass = 1 / mass

        self.speed_damping = speed_damping
        self.force_gain = force_gain
        self.angular_speed_gain = angular_speed_gain
        self.time_step = time_step

    @property
    def dt(self):
        return self.time_step

    @property
    def m(self):
        return self.mass

    @property
    def im(self):
        return self.inv_mass

    @property
    def v(self):
        return self.speed_damping

    @property
    def kf(self):
        return self.force_gain

    @property
    def kw(self):
        return self.angular_speed_gain
