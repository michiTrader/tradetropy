// Detect a user-driven Y range change and lock the autoscale.
//
// The autoscale is the sole programmatic authority over y_range. Whenever it
// writes the range it raises ylock.scaling so this watcher can tell its own
// writes apart from a genuine user gesture (ywheel_zoom / ypan / box_zoom).
// A change that arrives while scaling is false is therefore user-driven: we set
// locked=true so the autoscale suspends until the ResetTool clears it.

if (ylock.data['scaling'][0]) {
    return;
}
ylock.data['locked'][0] = true;
