use std::time::Duration;

pub struct ScrollTracker {
    tick: f64,
    last: f64,
    acc: f64,
}

const WHEEL_ACCEL_TIMEOUT: Duration = Duration::from_millis(100);
const WHEEL_ACCEL_SAMPLE_COUNT: usize = 3;
const WHEEL_ACCEL_MAX_GAIN: f64 = 8.0;
const WHEEL_ACCEL_CURVE_GAIN: f64 = 11.5;
const WHEEL_ACCEL_CURVE_SCALE: f64 = 12.0;

#[derive(Default)]
pub struct WheelAcceleration {
    samples: [(Duration, f64); WHEEL_ACCEL_SAMPLE_COUNT],
    sample_count: usize,
    modifiers: Option<u8>,
}

impl WheelAcceleration {
    pub fn apply(&mut self, time: Duration, steps: i8, modifiers: u8) -> f64 {
        let steps = f64::from(steps);
        let continues = self.sample_count > 0
            && self.modifiers == Some(modifiers)
            && self.samples[self.sample_count - 1].1.is_sign_positive() == steps.is_sign_positive()
            && time
                .checked_sub(self.samples[self.sample_count - 1].0)
                .is_some_and(|elapsed| elapsed <= WHEEL_ACCEL_TIMEOUT);

        if !continues {
            self.reset();
            self.modifiers = Some(modifiers);
        }

        let accelerated = steps * self.gain(time, steps);
        self.push_sample(time, steps);
        accelerated
    }

    fn gain(&self, time: Duration, steps: f64) -> f64 {
        if self.sample_count < WHEEL_ACCEL_SAMPLE_COUNT {
            return 1.0;
        }

        let Some(elapsed) = time.checked_sub(self.samples[0].0) else {
            return 1.0;
        };
        let distance = steps.abs()
            + self.samples[1..WHEEL_ACCEL_SAMPLE_COUNT]
                .iter()
                .map(|(_, steps)| steps.abs())
                .sum::<f64>();
        let average_interval = elapsed.as_secs_f64() / distance;

        if average_interval >= WHEEL_ACCEL_TIMEOUT.as_secs_f64() {
            return 1.0;
        }

        (WHEEL_ACCEL_CURVE_GAIN * (1.0 + WHEEL_ACCEL_CURVE_SCALE * average_interval).powi(-3))
            .clamp(1.0, WHEEL_ACCEL_MAX_GAIN)
    }

    fn push_sample(&mut self, time: Duration, steps: f64) {
        if self.sample_count < WHEEL_ACCEL_SAMPLE_COUNT {
            self.samples[self.sample_count] = (time, steps);
            self.sample_count += 1;
        } else {
            self.samples.rotate_left(1);
            self.samples[WHEEL_ACCEL_SAMPLE_COUNT - 1] = (time, steps);
        }
    }

    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.modifiers = None;
    }
}

impl ScrollTracker {
    #[allow(clippy::new_without_default)]
    pub fn new(tick: i8) -> Self {
        Self {
            tick: f64::from(tick),
            last: 0.,
            acc: 0.,
        }
    }

    pub fn accumulate(&mut self, amount: f64) -> i8 {
        let changed_direction = (self.last > 0. && amount < 0.) || (self.last < 0. && amount > 0.);
        if changed_direction {
            self.acc = 0.
        }

        self.last = amount;
        self.acc += amount;

        let mut ticks = 0;
        if self.acc.abs() >= self.tick {
            let clamped = self.acc.clamp(-127. * self.tick, 127. * self.tick);
            ticks = (clamped as i16 / self.tick as i16) as i8;
            self.acc %= self.tick;
        }

        ticks
    }

    pub fn reset(&mut self) {
        self.last = 0.;
        self.acc = 0.;
    }
}
