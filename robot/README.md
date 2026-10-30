# ORION Robot Link

How ORION drives a robot. ORION's brain stays on the PC (and its GPU); the
robot runs a small C library that speaks **ORION Robot Link (ORL) v1** over
USB serial or the network.

| Folder | What it is |
|---|---|
| `orion_link/` | The C99 library (`orion_link.h`, `orion_link.c`) and its C++11 wrapper (`OrionLink.hpp`). No heap, no floating point. Also an Arduino library (`library.properties`). |
| `sim/` | A software robot (`orl_sim.c`) that speaks ORL on stdin/stdout, so ORION can be tried with no hardware. |
| `examples/OrionRover/` | A two-wheeled rover for an Arduino Uno, an L298N motor driver and an HC-SR04 distance sensor. |

ORION's side is `orion_core/robot_link.py`; his tools are `robot_control` and
`robot_firmware`.

## Safety, in the library itself

- **Watchdog.** If the robot hears nothing from ORION for 500 ms, the motors
  stop. They do not restart by themselves: ORION has to send a new drive
  command, and the robot reports the watchdog stop until he does.
- **Emergency stop.** It latches. Motion is refused until it is released.
- **Range checks.** An out-of-range command is refused, never clamped into
  something else.
- **ORION's rules.** Stop and emergency stop always work. Motion and flashing
  happen only when the owner asks, never from a background job, plugin or agent.

## Try it without a robot

Say "connect to the software robot", then "drive forward for two seconds".

## The rover

1. Wire it as the comment at the top of `OrionRover.ino` says.
2. Plug the Uno in by USB, then ask ORION: "list my robot boards" and
   "flash the OrionRover example to COM5".
3. "Connect to the robot on serial COM5", then "forward 40 percent for one second".

## The wire format

```
0xA5 0x5A | version | type | seq | len (u16 LE) | payload | CRC-16/CCITT-FALSE (u16 LE)
```

The CRC covers version through payload. Payloads are at most 160 bytes.
Frame types are listed in `orion_link.h`.

## Building

The toolchain lives in `%USERPROFILE%\ORION\toolchain`: Zig's clang for C/C++,
and Arduino CLI for the boards. The library compiles clean with
`-std=c99 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror` for
Windows, 32- and 64-bit Raspberry Pi, and the Arduino Uno.
