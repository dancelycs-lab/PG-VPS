"""Correct envelope contraction only after a geometry update actually applies."""
import numpy as np
from research_platform.envelope_calibration import EnvelopeCalibratedMPC
class ValidEnvelopeMPC(EnvelopeCalibratedMPC):
 def widths(self):
  width=np.full(len(self.geometry.bound),self.initial_width)
  if self.update_geometry:
   for record in self.calibration_records:
    if record['applied']:width[record['face']]=.001
  return width
