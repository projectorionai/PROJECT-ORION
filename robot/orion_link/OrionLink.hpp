// ORION Robot Link - C++ wrapper for a robot's controller (C++11, no heap).
//
// Works with any byte stream that has int available(), int read() and
// size_t write(const uint8_t*, size_t): Arduino's Serial, a WiFiClient,
// or a small adapter round a Linux file descriptor.
//
//   orion::Link<decltype(Serial)> link(Serial, ORL_CAP_DRIVE | ORL_CAP_DISTANCE, 0, "Rover");
//   void loop() {
//       const uint32_t now = millis();
//       link.poll(now);                  // ORION's commands, acknowledged
//       int16_t left, right;
//       link.wheels(now, left, right);   // watchdog, e-stop and time limits applied
//       setMotors(left, right);
//   }
#ifndef ORION_LINK_HPP
#define ORION_LINK_HPP

#include "orion_link.h"

namespace orion {

template <typename Stream>
class Link {
public:
    Link(Stream &io, uint8_t caps, uint8_t servos, const char *name,
         uint32_t watchdog_ms = ORL_DEFAULT_WATCHDOG_MS)
        : io_(io), seq_(0)
    {
        orl_parser_init(&parser_);
        orl_robot_init(&robot_, caps, servos, watchdog_ms);
        hello_.caps = caps;
        hello_.servo_count = servos;
        uint8_t i = 0;
        for (; name != 0 && name[i] != '\0' && i < ORL_NAME_MAX; ++i) {
            hello_.name[i] = name[i];
        }
        hello_.name[i] = '\0';
    }

    // Read every waiting byte; apply and acknowledge each frame from ORION.
    // A HEARTBEAT is not acknowledged (it would double the traffic); a
    // HELLO is sent back the first time ORION is heard.
    void poll(uint32_t now_ms)
    {
        while (io_.available() > 0) {
            const int byte = io_.read();
            if (byte < 0) {
                break;
            }
            if (orl_parser_feed(&parser_, static_cast<uint8_t>(byte)) != 1) {
                continue;
            }
            const uint8_t was_heard = robot_.heard;
            const uint8_t result = orl_robot_apply(&robot_, &parser_.frame, now_ms);
            if (!was_heard) {
                sendHello();
            }
            if (parser_.frame.type != ORL_HEARTBEAT) {
                uint8_t payload[2];
                const uint16_t len = orl_pack_ack(parser_.frame.seq, result, payload);
                send(ORL_ACK, payload, len);
            }
        }
    }

    // The wheel power to apply now.
    void wheels(uint32_t now_ms, int16_t &left, int16_t &right)
    {
        orl_robot_tick(&robot_, now_ms, &left, &right);
    }

    void sendHello()
    {
        uint8_t payload[2 + ORL_NAME_MAX];
        const uint16_t len = orl_pack_hello(&hello_, payload);
        send(ORL_HELLO, payload, len);
    }

    // Sensor readings; the e-stop and watchdog bits are added here.
    void sendSensors(orl_sensors sensors)
    {
        sensors.status = static_cast<uint8_t>(sensors.status | orl_robot_status(&robot_));
        uint8_t payload[14];
        const uint16_t len = orl_pack_sensors(&sensors, payload);
        send(ORL_SENSORS, payload, len);
    }

    void sendEvent(uint8_t code, const char *text)
    {
        uint8_t payload[1 + ORL_TEXT_MAX];
        uint16_t len = 1;
        payload[0] = code;
        for (; text != 0 && text[len - 1] != '\0' && len < 1 + ORL_TEXT_MAX; ++len) {
            payload[len] = static_cast<uint8_t>(text[len - 1]);
        }
        send(ORL_EVENT, payload, len);
    }

    orl_robot &state() { return robot_; }
    const orl_robot &state() const { return robot_; }
    uint32_t crcErrors() const { return parser_.crc_errors; }

private:
    void send(uint8_t type, const uint8_t *payload, uint16_t len)
    {
        uint8_t frame[ORL_MAX_FRAME];
        const size_t size = orl_encode(type, seq_++, payload, len, frame, sizeof(frame));
        if (size > 0) {
            io_.write(frame, size);
        }
    }

    Stream &io_;
    orl_parser parser_;
    orl_robot robot_;
    orl_hello hello_;
    uint8_t seq_;
};

}  // namespace orion

#endif  // ORION_LINK_HPP
