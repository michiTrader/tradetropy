// ResetTool -> re-enable the autoscale.
//
// Clearing the lock lets the next x_range change (or data update) recompute the
// Y range from scratch, so pressing Reset returns to automatic scaling after
// the user has manually zoomed / panned the Y axis.

ylock.data['locked'][0] = false;
