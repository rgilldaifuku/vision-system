/*
  Minimal servo tracker.

  Input over Serial (115200), newline-terminated:
    - "1" / "0"           -> detection on/off (fast path)
    - "DET:1" / "DET:0"   -> same as above
    - "ANGLE:90"          -> optional direct set

  Behavior:
    - While detection is ON, move servo at 5 deg/sec (configurable).
    - If no message arrives for TIMEOUT_MS, detection is forced OFF.
*/

#include <Servo.h>

// --------- User settings ----------
static const uint8_t SERVO_PIN = 9;
static const long BAUD = 115200;

static const float RATE_DEG_PER_SEC = 5.0f;  // 5 degrees per second
static const int MIN_ANGLE = 0;
static const int MAX_ANGLE = 180;

// +1 moves upward, -1 moves downward when DET:1
static const int DIRECTION = +1;

// If we stop hearing from the PC, disable detection automatically.
static const unsigned long TIMEOUT_MS = 700;
// ----------------------------------

Servo servo;

static bool detected = false;
static unsigned long lastMsgMs = 0;
static unsigned long lastUpdateMs = 0;
static float angle = 90.0f;  // starting angle

static char buf[24];
static uint8_t bufLen = 0;

static inline int clampAngle(int a) {
  if (a < MIN_ANGLE) return MIN_ANGLE;
  if (a > MAX_ANGLE) return MAX_ANGLE;
  return a;
}

static inline bool isDigit(char c) { return (c >= '0' && c <= '9'); }

static int parseInt(const char *p) {
  int v = 0;
  bool any = false;
  while (*p && isDigit(*p)) {
    any = true;
    v = (v * 10) + (*p - '0');
    ++p;
  }
  return any ? v : 0;
}

static void handleLine(const char *s) {
  // skip leading spaces
  while (*s == ' ' || *s == '\t') ++s;
  if (*s == 0) return;

  lastMsgMs = millis();

  // Fast path: "1" or "0"
  if (s[0] == '1' && s[1] == 0) {
    detected = true;
    return;
  }
  if (s[0] == '0' && s[1] == 0) {
    detected = false;
    return;
  }

  // DET:x
  if (s[0] == 'D' && s[1] == 'E' && s[2] == 'T' && s[3] == ':' && s[4]) {
    detected = (s[4] != '0');
    return;
  }

  // ANGLE:nnn
  if (s[0] == 'A' && s[1] == 'N' && s[2] == 'G' && s[3] == 'L' && s[4] == 'E' && s[5] == ':' && s[6]) {
    int a = clampAngle(parseInt(s + 6));
    angle = (float)a;
    servo.write(a);
    return;
  }
}

void setup() {
  Serial.begin(BAUD);
  servo.attach(SERVO_PIN);
  angle = (float)clampAngle((int)angle);
  servo.write((int)angle);

  lastMsgMs = millis();
  lastUpdateMs = millis();
}

void loop() {
  // Read newline-terminated commands into a small fixed buffer.
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[bufLen] = 0;
      handleLine(buf);
      bufLen = 0;
    } else {
      if (bufLen < (sizeof(buf) - 1)) {
        buf[bufLen++] = c;
      } else {
        // overflow -> reset buffer (drop line)
        bufLen = 0;
      }
    }
  }

  unsigned long nowMs = millis();

  // timeout safety
  if ((nowMs - lastMsgMs) > TIMEOUT_MS) {
    detected = false;
  }

  // Update servo at a steady rate
  float dt = (nowMs - lastUpdateMs) * 0.001f;
  if (dt >= 0.01f) {  // <=100 Hz
    lastUpdateMs = nowMs;

    if (detected) {
      angle += (float)DIRECTION * RATE_DEG_PER_SEC * dt;
      int a = clampAngle((int)(angle + 0.5f));
      angle = (float)a;
      servo.write(a);
    }
  }
}

