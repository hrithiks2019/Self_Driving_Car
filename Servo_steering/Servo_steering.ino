
#include <Servo_ESP32.h>

static const int servoPin = 12;
String inString = "";
Servo_ESP32 servo1;

void setup() {
    Serial.begin(115200);
    servo1.attach(servoPin);
}

void loop() {
  while (Serial.available() > 0) {
    int inChar = Serial.read();
    if (isDigit(inChar)) {
      inString += (char)inChar;
    }
    if (inChar == '\n') {
      servo1.write(inString.toInt());
      inString = "";
    }
  }
}
