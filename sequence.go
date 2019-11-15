package seqtree

// MakeOneHotSequence creates a sequence of Timesteps for
// a slice of one-hot values.
// The inputs at each timestep are the previous values,
// and the outputs are the current values.
// The final output is 0, and the initial input has no
// features set.
func MakeOneHotSequence(seq []int, outputSize, numFeatures int) *Timestep {
	res := &Timestep{
		Features: make([]bool, numFeatures),
		Output:   make([]float64, outputSize),
		Target:   make([]float64, outputSize),
	}
	for _, x := range seq {
		res.Target[x] = 1.0
		res.Next = &Timestep{
			Prev:     res,
			Features: make([]bool, numFeatures),
			Output:   make([]float64, outputSize),
			Target:   make([]float64, outputSize),
		}
		res.Next.Features[x] = true
		res = res.Next
	}
	res.Target[0] = 1.0
	return res
}

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

// PropagateLoss computes the total loss for the sequence
// and sets all of the Gradients accordingly.
func (t *Timestep) PropagateLoss() float64 {
	loss := SoftmaxLoss(t.Output, t.Target)
	t.Gradient = SoftmaxLossGrad(t.Output, t.Target)
	if t.Next != nil {
		loss += t.Next.PropagateLoss()
	}
	return loss
}
