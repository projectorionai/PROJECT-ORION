/* ORION Robot Link v1 - see orion_link.h. */
#include "orion_link.h"

#include <string.h>

enum {
    PARSE_SYNC0 = 0,
    PARSE_SYNC1,
    PARSE_HEADER,
    PARSE_PAYLOAD,
    PARSE_CRC
};

/* ── little-endian helpers ────────────────────────────────────────────── */

static void put_u16(uint8_t *at, uint16_t value)
{
    at[0] = (uint8_t)(value & 0xFFu);
    at[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static uint16_t get_u16(const uint8_t *at)
{
    return (uint16_t)((uint16_t)at[0] | (uint16_t)((uint16_t)at[1] << 8));
}

static void put_i16(uint8_t *at, int16_t value)
{
    put_u16(at, (uint16_t)value);
}

static int16_t get_i16(const uint8_t *at)
{
    uint16_t raw = get_u16(at);
    /* Two's complement without relying on implementation-defined casts. */
    return (raw & 0x8000u) ? (int16_t)(-(int32_t)((~raw & 0xFFFFu) + 1u)) : (int16_t)raw;
}

/* ── CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) ───────────────────── */

static uint16_t crc_update(uint16_t crc, uint8_t byte)
{
    uint32_t value = (uint32_t)crc ^ ((uint32_t)byte << 8);
    uint8_t bit;
    for (bit = 0; bit < 8u; ++bit) {
        value = (value & 0x8000u) ? ((value << 1) ^ 0x1021u) : (value << 1);
    }
    return (uint16_t)(value & 0xFFFFu);
}

uint16_t orl_crc16(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFFu;
    size_t i;
    for (i = 0; i < len; ++i) {
        crc = crc_update(crc, data[i]);
    }
    return crc;
}

/* ── frames ───────────────────────────────────────────────────────────── */

size_t orl_encode(uint8_t type, uint8_t seq, const uint8_t *payload, uint16_t len,
                  uint8_t *out, size_t out_size)
{
    size_t total = (size_t)ORL_HEADER_SIZE + (size_t)len + 2u;
    uint16_t crc;
    if (len > ORL_MAX_PAYLOAD || out == NULL || out_size < total || (len > 0u && payload == NULL)) {
        return 0u;
    }
    out[0] = ORL_SYNC0;
    out[1] = ORL_SYNC1;
    out[2] = ORL_VERSION;
    out[3] = type;
    out[4] = seq;
    put_u16(&out[5], len);
    if (len > 0u) {
        memcpy(&out[7], payload, len);
    }
    crc = orl_crc16(&out[2], (size_t)5u + (size_t)len);
    put_u16(&out[7 + len], crc);
    return total;
}

void orl_parser_init(orl_parser *parser)
{
    memset(parser, 0, sizeof(*parser));
    parser->state = PARSE_SYNC0;
}

static void restart(orl_parser *parser)
{
    parser->state = PARSE_SYNC0;
    parser->got = 0u;
    parser->need = 0u;
}

int orl_parser_feed(orl_parser *parser, uint8_t byte)
{
    switch (parser->state) {
    case PARSE_SYNC0:
        if (byte == ORL_SYNC0) {
            parser->state = PARSE_SYNC1;
        } else {
            parser->dropped++;
        }
        return 0;
    case PARSE_SYNC1:
        if (byte == ORL_SYNC1) {
            parser->state = PARSE_HEADER;
            parser->got = 0u;
        } else if (byte == ORL_SYNC0) {
            parser->dropped++;           /* A5 A5 5A: the second A5 may start the frame */
        } else {
            parser->dropped += 2u;
            restart(parser);
        }
        return 0;
    case PARSE_HEADER:
        parser->header[parser->got++] = byte;
        if (parser->got == 5u) {
            uint16_t len = get_u16(&parser->header[3]);
            if (parser->header[0] != ORL_VERSION || len > ORL_MAX_PAYLOAD) {
                /* Not a frame we can read: hunt for the next sync. */
                parser->dropped += 7u;
                restart(parser);
                return 0;
            }
            parser->frame.type = parser->header[1];
            parser->frame.seq = parser->header[2];
            parser->frame.len = len;
            parser->got = 0u;
            parser->need = len;
            parser->state = (len > 0u) ? PARSE_PAYLOAD : PARSE_CRC;
        }
        return 0;
    case PARSE_PAYLOAD:
        parser->frame.payload[parser->got++] = byte;
        if (parser->got == parser->need) {
            parser->got = 0u;
            parser->state = PARSE_CRC;
        }
        return 0;
    case PARSE_CRC: {
        uint16_t expected;
        uint16_t i;
        parser->crc_bytes[parser->got++] = byte;
        if (parser->got < 2u) {
            return 0;
        }
        expected = 0xFFFFu;
        for (i = 0; i < 5u; ++i) {
            expected = crc_update(expected, parser->header[i]);
        }
        for (i = 0; i < parser->frame.len; ++i) {
            expected = crc_update(expected, parser->frame.payload[i]);
        }
        restart(parser);
        if (get_u16(parser->crc_bytes) != expected) {
            parser->crc_errors++;
            return -1;
        }
        return 1;
    }
    default:
        restart(parser);
        return 0;
    }
}

/* ── payloads ─────────────────────────────────────────────────────────── */

uint16_t orl_pack_hello(const orl_hello *hello, uint8_t *payload)
{
    size_t name_len = 0u;
    while (name_len < ORL_NAME_MAX && hello->name[name_len] != '\0') {
        ++name_len;
    }
    payload[0] = hello->caps;
    payload[1] = hello->servo_count;
    memcpy(&payload[2], hello->name, name_len);
    return (uint16_t)(2u + name_len);
}

int orl_unpack_hello(const orl_frame *frame, orl_hello *out)
{
    size_t name_len;
    if (frame->type != ORL_HELLO || frame->len < 2u || frame->len > 2u + ORL_NAME_MAX) {
        return 0;
    }
    out->caps = frame->payload[0];
    out->servo_count = frame->payload[1];
    name_len = (size_t)frame->len - 2u;
    memcpy(out->name, &frame->payload[2], name_len);
    out->name[name_len] = '\0';
    return 1;
}

uint16_t orl_pack_sensors(const orl_sensors *sensors, uint8_t *payload)
{
    put_u16(&payload[0], sensors->battery_mv);
    put_u16(&payload[2], sensors->distance_mm);
    put_i16(&payload[4], sensors->accel_mg[0]);
    put_i16(&payload[6], sensors->accel_mg[1]);
    put_i16(&payload[8], sensors->accel_mg[2]);
    put_i16(&payload[10], sensors->gyro_z_decidps);
    payload[12] = sensors->bumpers;
    payload[13] = sensors->status;
    return 14u;
}

int orl_unpack_sensors(const orl_frame *frame, orl_sensors *out)
{
    if (frame->type != ORL_SENSORS || frame->len != 14u) {
        return 0;
    }
    out->battery_mv = get_u16(&frame->payload[0]);
    out->distance_mm = get_u16(&frame->payload[2]);
    out->accel_mg[0] = get_i16(&frame->payload[4]);
    out->accel_mg[1] = get_i16(&frame->payload[6]);
    out->accel_mg[2] = get_i16(&frame->payload[8]);
    out->gyro_z_decidps = get_i16(&frame->payload[10]);
    out->bumpers = frame->payload[12];
    out->status = frame->payload[13];
    return 1;
}

uint16_t orl_pack_drive(const orl_drive *drive, uint8_t *payload)
{
    put_i16(&payload[0], drive->left);
    put_i16(&payload[2], drive->right);
    put_u16(&payload[4], drive->duration_ms);
    return 6u;
}

int orl_unpack_drive(const orl_frame *frame, orl_drive *out)
{
    if (frame->type != ORL_DRIVE || frame->len != 6u) {
        return 0;
    }
    out->left = get_i16(&frame->payload[0]);
    out->right = get_i16(&frame->payload[2]);
    out->duration_ms = get_u16(&frame->payload[4]);
    return 1;
}

uint16_t orl_pack_servo(const orl_servo *servo, uint8_t *payload)
{
    payload[0] = servo->index;
    put_u16(&payload[1], servo->decidegrees);
    return 3u;
}

int orl_unpack_servo(const orl_frame *frame, orl_servo *out)
{
    if (frame->type != ORL_SERVO || frame->len != 3u) {
        return 0;
    }
    out->index = frame->payload[0];
    out->decidegrees = get_u16(&frame->payload[1]);
    return 1;
}

uint16_t orl_pack_ack(uint8_t acked_seq, uint8_t result, uint8_t *payload)
{
    payload[0] = acked_seq;
    payload[1] = result;
    return 2u;
}

/* ── the robot's side ─────────────────────────────────────────────────── */

void orl_robot_init(orl_robot *robot, uint8_t caps, uint8_t servo_count, uint32_t watchdog_ms)
{
    memset(robot, 0, sizeof(*robot));
    robot->caps = caps;
    robot->servo_count = servo_count;
    robot->watchdog_ms = watchdog_ms ? watchdog_ms : ORL_DEFAULT_WATCHDOG_MS;
}

static void halt(orl_robot *robot)
{
    robot->left = 0;
    robot->right = 0;
    robot->drive_duration_ms = 0u;
}

uint8_t orl_robot_apply(orl_robot *robot, const orl_frame *frame, uint32_t now_ms)
{
    /* Any valid frame from ORION feeds the watchdog. A trip stays reported
     * (ORL_STATUS_WATCHDOG) until ORION gives a fresh motion order, so he
     * can see why the robot stopped; the wheels never restart by themselves. */
    robot->last_heard_ms = now_ms;
    robot->heard = 1u;

    switch (frame->type) {
    case ORL_HEARTBEAT:
        return ORL_OK;
    case ORL_STOP:
        halt(robot);
        robot->watchdog_tripped = 0u;
        return ORL_OK;
    case ORL_ESTOP:
        if (frame->len != 1u || frame->payload[0] > 1u) {
            return ORL_REJECTED;
        }
        robot->estopped = frame->payload[0];
        halt(robot);                     /* engaging or releasing: start from still */
        return ORL_OK;
    case ORL_DRIVE: {
        orl_drive drive;
        if (!(robot->caps & ORL_CAP_DRIVE)) {
            return ORL_UNSUPPORTED;
        }
        if (!orl_unpack_drive(frame, &drive)
            || drive.left < -ORL_DRIVE_LIMIT || drive.left > ORL_DRIVE_LIMIT
            || drive.right < -ORL_DRIVE_LIMIT || drive.right > ORL_DRIVE_LIMIT) {
            return ORL_REJECTED;
        }
        if (robot->estopped) {
            halt(robot);
            return ORL_ESTOPPED;
        }
        robot->left = drive.left;
        robot->right = drive.right;
        robot->drive_start_ms = now_ms;
        robot->drive_duration_ms = drive.duration_ms;
        robot->watchdog_tripped = 0u;
        return ORL_OK;
    }
    case ORL_SERVO: {
        orl_servo servo;
        if (!(robot->caps & ORL_CAP_SERVO)) {
            return ORL_UNSUPPORTED;
        }
        if (!orl_unpack_servo(frame, &servo) || servo.index >= robot->servo_count
            || servo.decidegrees > ORL_SERVO_MAX_DECIDEG) {
            return ORL_REJECTED;
        }
        if (robot->estopped) {
            return ORL_ESTOPPED;
        }
        robot->servo_index = servo.index;
        robot->servo_target = servo.decidegrees;
        robot->servo_pending = 1u;
        return ORL_OK;
    }
    case ORL_LED:
        if (!(robot->caps & ORL_CAP_LED)) {
            return ORL_UNSUPPORTED;
        }
        if (frame->len != 3u) {
            return ORL_REJECTED;
        }
        memcpy(robot->led, frame->payload, 3u);
        robot->led_pending = 1u;
        return ORL_OK;
    case ORL_SAY:
        if (!(robot->caps & ORL_CAP_SPEAKER)) {
            return ORL_UNSUPPORTED;
        }
        if (frame->len == 0u || frame->len > ORL_TEXT_MAX) {
            return ORL_REJECTED;
        }
        memcpy(robot->say, frame->payload, frame->len);
        robot->say[frame->len] = '\0';
        robot->say_pending = 1u;
        return ORL_OK;
    default:
        return ORL_UNSUPPORTED;
    }
}

void orl_robot_tick(orl_robot *robot, uint32_t now_ms, int16_t *left, int16_t *right)
{
    /* Unsigned subtraction is wrap-safe: it is the elapsed time even when
     * the millisecond counter rolled over between the two readings. */
    if (!robot->heard || (uint32_t)(now_ms - robot->last_heard_ms) > robot->watchdog_ms) {
        if (robot->heard) {
            robot->watchdog_tripped = 1u;
        }
        halt(robot);
    }
    if (robot->estopped) {
        halt(robot);
    }
    if (robot->drive_duration_ms != 0u
        && (uint32_t)(now_ms - robot->drive_start_ms) >= (uint32_t)robot->drive_duration_ms) {
        halt(robot);
    }
    *left = robot->left;
    *right = robot->right;
}

uint8_t orl_robot_status(const orl_robot *robot)
{
    uint8_t status = 0u;
    if (robot->estopped) {
        status |= ORL_STATUS_ESTOPPED;
    }
    if (robot->watchdog_tripped) {
        status |= ORL_STATUS_WATCHDOG;
    }
    return status;
}
