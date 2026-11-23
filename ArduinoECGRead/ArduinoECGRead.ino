#include <Arduino.h>

constexpr uint8_t ECG_PIN = A0;
constexpr unsigned long SAMPLE_INTERVAL_US = 1000;
unsigned long nextSampleUs;

void setup() {
  pinMode(ECG_PIN, INPUT);
  Serial.begin(115200);
  nextSampleUs = micros();
}

void loop() {
  const unsigned long nowUs = micros();
  if (static_cast<long>(nowUs - nextSampleUs) < 0) {
    return;
  }

  nextSampleUs += SAMPLE_INTERVAL_US;
  // Skip missed intervals so a delay cannot trigger a burst of samples.
  if (static_cast<long>(nowUs - nextSampleUs) >= 0) {
    nextSampleUs = nowUs + SAMPLE_INTERVAL_US;
  }

  const int adcCount = analogRead(ECG_PIN);
  Serial.println(adcCount);
  Serial.println(analogRead(A0));
}
