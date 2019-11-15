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
		Output:   make([]float32, outputSize),
		Target:   make([]float32, outputSize),
	}
	start := res
	for _, x := range seq {
		res.Target[x] = 1.0
		res.Next = &Timestep{
			Prev:     res,
			Features: make([]bool, numFeatures),
			Output:   make([]float32, outputSize),
			Target:   make([]float32, outputSize),
		}
		res.Next.Features[x] = true
		res = res.Next
	}
	res.Target[0] = 1.0
	return start
}

// AllTimesteps gets all of the timesteps from all of the
// sequences, given a list of sequence starts.
func AllTimesteps(starts ...*Timestep) []*Timestep {
	var res []*Timestep
	for _, seq := range starts {
		seq.Iterate(func(t *Timestep) {
			res = append(res, t)
		})
	}
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
	Output []float32

	// The following two fields are used during training.
	// The Target indicates the ground-truth label, and
	// the Gradient indicates the gradient of the loss
	// function with respect to the outputs.
	Target   []float32
	Gradient []float32
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
func (t *Timestep) PropagateLoss() float32 {
	loss := SoftmaxLoss(t.Output, t.Target)
	t.Gradient = SoftmaxLossGrad(t.Output, t.Target)
	if t.Next != nil {
		loss += t.Next.PropagateLoss()
	}
	return loss
}
