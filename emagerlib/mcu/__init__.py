"""Host-side support for running a model on a microcontroller.

The PC does no signal processing here: it forwards the base station's raw USB
byte stream to the MCU untouched (:mod:`emagerlib.mcu.relay`) and reads back the
predictions the MCU emits (:mod:`emagerlib.mcu.protocol`,
:mod:`emagerlib.mcu.prediction_reader`).

Everything downstream of the relay -- frame decoding, filtering, feature
extraction, inference -- lives in the firmware, in a separate repository. The
wire format the firmware must implement is specified in `docs/MCU_PROTOCOL.md`.
"""
