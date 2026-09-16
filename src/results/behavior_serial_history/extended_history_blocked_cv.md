# Extended Serial-History Validation

Primary question: do serial-history models improve pooled blocked prediction beyond the current-input baseline?

- Subjects: 24
- Median common trials per subject: 666
- Blocks per day: 6; purge: 5 raw trials
- Predeclared boundary-jitter seed: 20260808 (subject offsets of 100)

## Delta R2 from current-input baseline

- lag1: median R/T = +0.000588 / -0.000387; one-sided Holm p = 1 / 1; positive subjects = 14/11
- lag1_to_5: median R/T = -0.004208 / -0.004828; one-sided Holm p = 1 / 1; positive subjects = 5/4
- exp_all: median R/T = -0.001955 / -0.000832; one-sided Holm p = 1 / 1; positive subjects = 9/10
- exp_drawing: median R/T = +0.001319 / -0.000246; one-sided Holm p = 1 / 1; positive subjects = 13/11

## Incremental delta R2 beyond lag 1

- lag1_to_5: median R/T = -0.004947 / -0.009811; Holm p = 0.000811578 / 0.000109292
- exp_all: median R/T = -0.001899 / -0.002798; Holm p = 0.0439801 / 0.0439801
- exp_drawing: median R/T = -0.000743 / +0.000784; Holm p = 1 / 1

## Simultaneous unique lag coefficients

- Lag 1: H_RR=+0.0385, H_RT=+0.0069, H_TR=-0.0064, H_TT=+0.0898
- Lag 2: H_RR=+0.0042, H_RT=-0.0077, H_TR=-0.0051, H_TT=+0.0076
- Lag 3: H_RR=-0.0001, H_RT=-0.0091, H_TR=+0.0038, H_TT=+0.0046
- Lag 4: H_RR=+0.0044, H_RT=+0.0019, H_TR=-0.0073, H_TT=-0.0052
- Lag 5: H_RR=+0.0030, H_RT=-0.0114, H_TR=-0.0072, H_TT=+0.0135

## Nested exponential lambda

- exp_all: median 0.305, IQR [0.1325, 0.6], boundary subjects 1
- exp_drawing: median 0.312, IQR [0.08, 0.6124999999999999], boundary subjects 5
