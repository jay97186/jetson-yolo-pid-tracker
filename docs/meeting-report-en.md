# April 16 Meeting Report: English Summary

Source: [24-slide Meeting Report](https://docs.google.com/presentation/d/1fAQqj434E0ylYbD4rZnKBL8-_9kuzTRB_qsPAZNE20o/edit). The cover states April 16 without a year. This summary follows the presentation's structure and distinguishes reported experiments from the supplied implementation.

## Progress (slide 2)

The team resolved package incompatibilities that had prevented the code from running. The report describes successful servo control on Jetson Orin Nano, real-time horizontal face tracking using PID, and a comparison of controller gains. The setup slide specifies I2C control through PCA9685.

## Research objective (slide 3)

Underwater free-space optical communication requires accurate alignment. The proposed application uses underwater imagery to recognize ships and align a seabed base station with an LED communication point on a ship. Face tracking serves as the current tracking demonstration. The report does not claim underwater validation.

## Experimental setup (slide 4)

Jetson Orin Nano processes CSI camera frames and controls a servo through a PCA9685 module using I2C and PWM. The servo rotates the camera toward the target. The presentation names YOLOv8 and `face.pt`. The supplied implementation instead points to `/home/jetson/yolov8n-face.pt`; no model file accompanies the source folder.

## Feedback control (slides 5–14)

The image center is the setpoint. The target's bounding-box center supplies the position measurement, and the controller updates the servo angle until the error falls within the dead zone.

The animated examples use `error = setpoint - position` and illustrate incremental proportional control, including example angle changes of 50° and 20°. These are explanatory examples, not the actual deployed gain values. The provided code defines pixel error as `detection_center_x - FRAME_CX`, includes integral and derivative terms, and exposes a direction-reversal switch.

The presentation describes comparing three PID configurations and starting recording with `r`. The archived Python file implements quitting with `q` or Esc but contains no recording handler.

## Digital filtering and demonstrations (slides 15–22)

The report introduces exponential moving average (EMA) and moving average filtering, followed by unfiltered and filtered demonstrations. The labeled filtered examples use:

| Example | Dead zone | Duration |
| --- | --- | --- |
| First filtered example | 40 pixels | 5 seconds |
| Second filtered example | 10 pixels | 5 seconds |
| Third filtered example | 10 pixels | 12 seconds |

The provided tracker does not implement these measurement filters. Its configured dead zone is 80 pixels. The source includes two control-flow diagrams but no raw time-series data, filter coefficients, or complete configuration records for each video.

## Conclusion (slide 23)

| Proportional gain | Reported behavior |
| --- | --- |
| `Kp = 0.001` | Slower transient response |
| `Kp = 0.003` | Better balance of speed and stability |
| `Kp = 0.005` | Fastest approach to the setpoint, with persistent oscillation and no stable convergence |

These conclusions apply to the reported setup. The slides do not supply numerical settling-time or overshoot measurements. The archived source uses `KP = 0.003`, `KI = 0.005`, and `KD = 0.0001`.

## Future plan (slide 24)

The next steps are two-axis PID control, a model for rolling-shutter patterns or ship recognition, and real-time pattern tracking in free space. The supplied code remains a single-axis face-tracking prototype.
