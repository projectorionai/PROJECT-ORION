/*
 * ORION Robot Link (ORL) v1 - the wire protocol between ORION (on the PC)
 * and a robot's own controller (Arduino, ESP32, Raspberry Pi, ...).
 *
 * Portable C99: no heap, no floating point, no platform headers. Every
 * multi-byte field is little-endian. The Python side is
 * orion_core/robot_link.py; both are tested against each other.
 *
 * A frame:
 *
 *   0xA5 0x5A | version | type | seq | len (u16) | payload[len] | crc (u16)
 *
 * The CRC is CRC-16/CCITT-FALSE over version..payload. A payload is at most
 * ORL_MAX_PAYLOAD bytes.
 *
 * Safety lives here, not in each robot's sketch (orl_robot_*):
 *   - a watchdog: no valid frame from ORION for watchdog_ms and the motors
 *     stop until he is heard again;
 *   - an emergency stop that latches: motion is refused until released;
 *   - every command is range-checked; a bad one is refused, never clamped
 *     into something the sender did not ask for.
 */
#ifndef ORION_LINK_H
#define ORION_LINK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORL_VERSION 1u
#define ORL_SYNC0 0xA5u
#define ORL_SYNC1 0x5Au
#define ORL_MAX_PAYLOAD 160u
#define ORL_HEADER_SIZE 7u               /* sync x2, version, type, seq, len x2 */
#define ORL_MAX_FRAME (ORL_HEADER_SIZE + ORL_MAX_PAYLOAD + 2u)
#define ORL_DEFAULT_WATCHDOG_MS 500u

/* Frame types: robot -> ORION */
#define ORL_HELLO 0x01u
#define ORL_SENSORS 0x02u
#define ORL_ACK 0x03u
#define ORL_EVENT 0x04u
/* Frame types: ORION -> robot */
#define ORL_HEARTBEAT 0x10u
#define ORL_DRIVE 0x11u
#define ORL_SERVO 0x12u
#define ORL_STOP 0x13u
#define ORL_ESTOP 0x14u
#define ORL_LED 0x15u
#define ORL_SAY 0x16u

/* ACK results */
#define ORL_OK 0u
#define ORL_REJECTED 1u                  /* out of range or malformed */
#define ORL_UNSUPPORTED 2u               /* this robot has no such part */
#define ORL_ESTOPPED 3u                  /* motion while the e-stop is engaged */

/* HELLO capability bits */
#define ORL_CAP_DRIVE 0x01u
#define ORL_CAP_SERVO 0x02u
#define ORL_CAP_DISTANCE 0x04u
#define ORL_CAP_BATTERY 0x08u
#define ORL_CAP_IMU 0x10u
#define ORL_CAP_SPEAKER 0x20u
#define ORL_CAP_LED 0x40u

/* SENSORS status bits */
#define ORL_STATUS_ESTOPPED 0x01u
#define ORL_STATUS_WATCHDOG 0x02u
#define ORL_STATUS_LOW_BATTERY 0x04u

#define ORL_DRIVE_LIMIT 1000             /* wheel power, per mille, either way */
#define ORL_SERVO_MAX_DECIDEG 1800u      /* 0.0 .. 180.0 degrees */
#define ORL_DISTANCE_UNKNOWN 0xFFFFu
#define ORL_NAME_MAX 32u
#define ORL_TEXT_MAX 128u

typedef struct {
    uint8_t type;
    uint8_t seq;
    uint16_t len;
    uint8_t payload[ORL_MAX_PAYLOAD];
} orl_frame;

typedef struct {
    uint8_t caps;
    uint8_t servo_count;
    char name[ORL_NAME_MAX + 1u];        /* NUL-terminated */
} orl_hello;

typedef struct {
    uint16_t battery_mv;
    uint16_t distance_mm;                /* ORL_DISTANCE_UNKNOWN when there is none */
    int16_t accel_mg[3];
    int16_t gyro_z_decidps;
    uint8_t bumpers;
    uint8_t status;                      /* ORL_STATUS_* */
} orl_sensors;

typedef struct {
    int16_t left;                        /* -1000 .. 1000 */
    int16_t right;
    uint16_t duration_ms;                /* 0: until the next command or the watchdog */
} orl_drive;

typedef struct {
    uint8_t index;
    uint16_t decidegrees;                /* 0 .. 1800 */
} orl_servo;

/* ── CRC and frames ───────────────────────────────────────────────────── */

uint16_t orl_crc16(const uint8_t *data, size_t len);

/* Encode one frame into out. Returns the bytes written, or 0 when the
 * payload is too long or out is too small. */
size_t orl_encode(uint8_t type, uint8_t seq, const uint8_t *payload, uint16_t len,
                  uint8_t *out, size_t out_size);

typedef struct {
    uint8_t state;
    uint16_t got;
    uint16_t need;
    uint8_t header[5];                   /* version, type, seq, len x2 */
    uint8_t crc_bytes[2];
    orl_frame frame;
    uint32_t crc_errors;
    uint32_t dropped;                    /* bytes skipped while hunting for a frame */
} orl_parser;

void orl_parser_init(orl_parser *parser);

/* Feed one received byte. Returns 1 when parser->frame holds a complete,
 * CRC-checked frame, -1 when a frame failed its CRC (the parser has already
 * resynchronised), and 0 otherwise. */
int orl_parser_feed(orl_parser *parser, uint8_t byte);

/* ── payloads ─────────────────────────────────────────────────────────── */

uint16_t orl_pack_hello(const orl_hello *hello, uint8_t *payload);
int orl_unpack_hello(const orl_frame *frame, orl_hello *out);
uint16_t orl_pack_sensors(const orl_sensors *sensors, uint8_t *payload);
int orl_unpack_sensors(const orl_frame *frame, orl_sensors *out);
uint16_t orl_pack_drive(const orl_drive *drive, uint8_t *payload);
int orl_unpack_drive(const orl_frame *frame, orl_drive *out);
uint16_t orl_pack_servo(const orl_servo *servo, uint8_t *payload);
int orl_unpack_servo(const orl_frame *frame, orl_servo *out);
uint16_t orl_pack_ack(uint8_t acked_seq, uint8_t result, uint8_t *payload);

/* ── the robot's side: commands, watchdog and e-stop ──────────────────── */

typedef struct {
    uint8_t caps;
    uint8_t servo_count;
    uint32_t watchdog_ms;
    uint32_t last_heard_ms;
    uint8_t heard;                       /* 1 once ORION has been heard at all */
    uint8_t estopped;
    uint8_t watchdog_tripped;
    int16_t left;
    int16_t right;
    uint32_t drive_start_ms;
    uint16_t drive_duration_ms;          /* 0: no time limit */
    uint8_t servo_pending;               /* 1 when servo_index/servo_target are new */
    uint8_t servo_index;
    uint16_t servo_target;
    uint8_t led[3];
    uint8_t led_pending;
    char say[ORL_TEXT_MAX + 1u];
    uint8_t say_pending;
} orl_robot;

void orl_robot_init(orl_robot *robot, uint8_t caps, uint8_t servo_count, uint32_t watchdog_ms);

/* Apply one frame from ORION at now_ms. Returns the ORL_OK / ORL_REJECTED /
 * ORL_UNSUPPORTED / ORL_ESTOPPED result to acknowledge it with. */
uint8_t orl_robot_apply(orl_robot *robot, const orl_frame *frame, uint32_t now_ms);

/* Call every loop. Writes the wheel power to drive now (zero when the
 * e-stop is engaged, the watchdog has tripped or the drive's time is up).
 * Safe across the 32-bit millisecond wrap. */
void orl_robot_tick(orl_robot *robot, uint32_t now_ms, int16_t *left, int16_t *right);

/* The SENSORS status bits this state implies. */
uint8_t orl_robot_status(const orl_robot *robot);

#ifdef __cplusplus
}
#endif

#endif /* ORION_LINK_H */
