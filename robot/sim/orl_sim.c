/*
 * A software robot that speaks ORION Robot Link over stdin/stdout, built on
 * the same orion_link.c a real robot uses - so ORION can drive a robot
 * before there is one, and the C and Python sides are tested against each
 * other end to end.
 *
 * Time is ORION's: each HEARTBEAT carries his clock (u32 ms), and that is
 * the simulator's "now" - so the watchdog and timed drives behave exactly as
 * on a board, and a test can replay them deterministically.
 *
 * The robot: two wheels, full power = 300 mm/s, a wall 2000 mm ahead, a
 * 7.4 V battery, two servos, an LED and a speaker.
 *
 *   zig cc -std=c99 -O2 -I ../orion_link orl_sim.c ../orion_link/orion_link.c -o orl_sim.exe
 */
#include <stdio.h>
#include <string.h>

#include "orion_link.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define SIM_WALL_MM INT64_C(2000)
#define SIM_FULL_SPEED_MM_S INT64_C(300)

static uint8_t seq_out = 0u;

static void send_frame(uint8_t type, const uint8_t *payload, uint16_t len)
{
    uint8_t out[ORL_MAX_FRAME];
    size_t size = orl_encode(type, seq_out++, payload, len, out, sizeof(out));
    if (size > 0u) {
        fwrite(out, 1u, size, stdout);
        fflush(stdout);
    }
}

static uint32_t read_u32(const uint8_t *at)
{
    return (uint32_t)at[0] | ((uint32_t)at[1] << 8) | ((uint32_t)at[2] << 16) | ((uint32_t)at[3] << 24);
}

int main(void)
{
    orl_parser parser;
    orl_robot robot;
    orl_hello hello;
    uint32_t now = 0u;
    uint32_t last = 0u;
    int64_t travelled_um = 0;            /* micrometres, so slow drives still move */
    int byte;

#ifdef _WIN32
    (void)_setmode(_fileno(stdin), _O_BINARY);
    (void)_setmode(_fileno(stdout), _O_BINARY);
#endif
    orl_parser_init(&parser);
    orl_robot_init(&robot, (uint8_t)(ORL_CAP_DRIVE | ORL_CAP_SERVO | ORL_CAP_DISTANCE | ORL_CAP_BATTERY
                                     | ORL_CAP_LED | ORL_CAP_SPEAKER), 2u, ORL_DEFAULT_WATCHDOG_MS);
    memset(&hello, 0, sizeof(hello));
    hello.caps = robot.caps;
    hello.servo_count = robot.servo_count;
    memcpy(hello.name, "ORION Sim Rover", 15u);

    while ((byte = getchar()) != EOF) {
        uint8_t payload[ORL_MAX_PAYLOAD];
        uint8_t was_heard;
        uint8_t result;
        int16_t left = 0;
        int16_t right = 0;
        if (orl_parser_feed(&parser, (uint8_t)byte) != 1) {
            continue;
        }
        if (parser.frame.type == ORL_HEARTBEAT && parser.frame.len == 4u) {
            now = read_u32(parser.frame.payload);
        }
        /* Move for the time since the last frame with the wheels as they
         * were - checked against the watchdog and any time limit first. */
        orl_robot_tick(&robot, now, &left, &right);
        travelled_um += ((int64_t)left + (int64_t)right) / 2 * SIM_FULL_SPEED_MM_S
                        * (int64_t)(uint32_t)(now - last) / 1000;
        last = now;

        was_heard = robot.heard;
        result = orl_robot_apply(&robot, &parser.frame, now);
        if (!was_heard) {
            uint16_t len = orl_pack_hello(&hello, payload);
            send_frame(ORL_HELLO, payload, len);
        }
        if (parser.frame.type == ORL_HEARTBEAT) {
            orl_sensors sensors;
            int64_t distance = SIM_WALL_MM - travelled_um / 1000;
            memset(&sensors, 0, sizeof(sensors));
            sensors.battery_mv = 7400u;
            sensors.distance_mm = (uint16_t)(distance < 0 ? 0 : (distance > 60000 ? 60000 : distance));
            sensors.accel_mg[2] = 1000;
            sensors.status = orl_robot_status(&robot);
            send_frame(ORL_SENSORS, payload, orl_pack_sensors(&sensors, payload));
        } else {
            send_frame(ORL_ACK, payload, orl_pack_ack(parser.frame.seq, result, payload));
        }
    }
    return 0;
}
