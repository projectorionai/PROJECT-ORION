// A two-wheeled rover that ORION drives over USB (ORION Robot Link v1).
//
// Wiring (Arduino Uno + L298N motor driver + HC-SR04 distance sensor):
//   L298N ENA -> D5 (PWM)   IN1 -> D7   IN2 -> D8     (left motor)
//   L298N ENB -> D6 (PWM)   IN3 -> D9   IN4 -> D10    (right motor)
//   HC-SR04 TRIG -> D2      ECHO -> D3
//   Battery divider (2:1) -> A0
//
// ORION connects with:  robot_control connect address="serial:COM5"
// The library stops the motors within 500 ms of losing ORION's heartbeat,
// and refuses motion while ORION's emergency stop is engaged.
#include <OrionLink.hpp>

namespace {

const uint8_t LEFT_PWM = 5, LEFT_A = 7, LEFT_B = 8;
const uint8_t RIGHT_PWM = 6, RIGHT_A = 9, RIGHT_B = 10;
const uint8_t TRIG = 2, ECHO = 3;
const uint8_t BATTERY = A0;
const uint16_t LOW_BATTERY_MV = 6400;

// decltype(Serial): HardwareSerial on the Uno, Nano and Mega, the USB Serial_
// class on the Leonardo and Micro - one sketch for all of them.
orion::Link<decltype(Serial)> link(Serial, ORL_CAP_DRIVE | ORL_CAP_DISTANCE | ORL_CAP_BATTERY,
                                   0, "OrionRover");
uint32_t lastReport = 0;

void motor(uint8_t pwm, uint8_t a, uint8_t b, int16_t power) {
    // power is -1000..1000 per mille.
    const int32_t duty = static_cast<int32_t>(power < 0 ? -power : power) * 255 / 1000;
    digitalWrite(a, power > 0 ? HIGH : LOW);
    digitalWrite(b, power < 0 ? HIGH : LOW);
    analogWrite(pwm, static_cast<int>(duty));
}

uint16_t distanceMm() {
    digitalWrite(TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    const unsigned long echo = pulseIn(ECHO, HIGH, 25000UL);   // ~4 m at most
    if (echo == 0) {
        return ORL_DISTANCE_UNKNOWN;
    }
    return static_cast<uint16_t>(echo * 343UL / 2000UL);       // speed of sound, there and back
}

uint16_t batteryMv() {
    // 2:1 divider into a 5 V, 10-bit ADC.
    return static_cast<uint16_t>(static_cast<uint32_t>(analogRead(BATTERY)) * 5000UL * 2UL / 1023UL);
}

}  // namespace

void setup() {
    const uint8_t outputs[] = {LEFT_PWM, LEFT_A, LEFT_B, RIGHT_PWM, RIGHT_A, RIGHT_B, TRIG};
    for (uint8_t pin : outputs) {
        pinMode(pin, OUTPUT);
    }
    pinMode(ECHO, INPUT);
    motor(LEFT_PWM, LEFT_A, LEFT_B, 0);
    motor(RIGHT_PWM, RIGHT_A, RIGHT_B, 0);
    Serial.begin(115200);
}

void loop() {
    const uint32_t now = millis();
    link.poll(now);
    int16_t left = 0, right = 0;
    link.wheels(now, left, right);
    motor(LEFT_PWM, LEFT_A, LEFT_B, left);
    motor(RIGHT_PWM, RIGHT_A, RIGHT_B, right);
    if (now - lastReport >= 100) {
        lastReport = now;
        orl_sensors sensors = {};
        sensors.battery_mv = batteryMv();
        sensors.distance_mm = distanceMm();
        sensors.status = sensors.battery_mv < LOW_BATTERY_MV ? ORL_STATUS_LOW_BATTERY : 0;
        link.sendSensors(sensors);
    }
}
