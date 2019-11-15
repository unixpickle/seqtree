package seqtree

// Timestep represents a single timestep in a linked-list
// representing a sequence.
type Timestep struct {
	// Prev is the previous timestep.
	// It is nil at the start of the sequence.
	Prev *Timestep

	// Next is the next timestep.
	// It is nil at the end of the sequence.
	Next *Timestep

	// Features stores the current feature bitmap.
	Features []bool

	// Output is the current prediction parameter vector
	// for this timestamp.
	Output []float64

	// The following two fields are used during training.
	// The Target indicates the ground-truth label, and
	// the Gradient indicates the gradient of the loss
	// function with respect to the outputs.
	Target   []float64
	Gradient []float64
}

// Iterate iterates over the timesteps of this model, from
// first to last.
func (t *Timestep) Iterate(f func(t *Timestep)) {
	ts := t
	for ts != nil {
		f(ts)
		ts = ts.Next
	}
}

// BranchFeature computes the value of the feature, which
// may be in the past.
func (t *Timestep) BranchFeature(b BranchFeature) bool {
	ts := t
	for i := 0; i < b.StepsInPast; i++ {
		if ts != nil {
			ts = ts.Prev
		}
	}
	if ts == nil {
		return b.Feature == -1
	}
	return ts.Features[b.Feature]
}
