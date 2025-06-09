#define dirPin 4
#define stepPin 3

// Parameter mekanik
#define stepsPerRevolution 3200     // 200 x 1/16 microstep
#define leadScrewPitch 1.5          // mm per rev (M12 x 1.5)
#define targetDistanceMM 10         // Jarak naik dalam mm
#define movementTimeSeconds 2.0     // Waktu yang diinginkan (detik)

// Perhitungan otomatis
float mmPerStep = leadScrewPitch / stepsPerRevolution;
long totalSteps = targetDistanceMM / mmPerStep;
float stepsPerSecond = totalSteps / movementTimeSeconds;
int delayUs = 1000000.0 / stepsPerSecond;

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(9600);

  digitalWrite(dirPin, HIGH); // Gerak ke atas

  Serial.print("Target Naik: ");
  Serial.print(targetDistanceMM);
  Serial.print(" mm dalam ");
  Serial.print(movementTimeSeconds);
  Serial.println(" detik");

  Serial.print("Total step: ");
  Serial.println(totalSteps);

  Serial.print("Delay per step (us): ");
  Serial.println(delayUs);

  // Gerakkan stepper
  for (long i = 0; i < totalSteps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(delayUs);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(delayUs);
  }

  Serial.println("Gerakan selesai.");
}

void loop() {
  // Kosongin aja, program cuma jalan sekali di setup
}
