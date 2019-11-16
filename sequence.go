package seqtree

// MakeOneHotSequence creates a sequence for a slice of
// one-hot values.
// The inputs at each timestep are the previous values,
// and the outputs are the current values.
// The final output is 0, and the initial input has no
// features set.
func MakeOneHotSequence(seq []int, outputSize, numFeatures int) Sequence {
	ts := &Timestep{
		Features: make([]bool, numFeatures),
		Output:   make([]float32, outputSize),
		Target:   make([]float32, outputSize),
	}
	res := Sequence{}
	for _, x := range seq {
		ts.Target[x] = 1.0
		res = append(res, ts)
		ts = &Timestep{
			Features: make([]bool, numFeatures),
			Output:   make([]float32, outputSize),
			Target:   make([]float32, outputSize),
		}
		ts.Features[x] = true
	}
	ts.Target[0] = 1.0
	return append(res, ts)
}

// AllTimesteps gets all of the timestep samples from all
// of the sequences.
func AllTimesteps(seqs ...Sequence) []*TimestepSample {
	var res []*TimestepSample
	for _, seq := range seqs {
		for i := range seq {
			res = append(res, &TimestepSample{Sequence: seq, Index: i})
		}
	}
	return res
}

type Sequence []*Timestep

// PropagateLoss computes the mean loss for the sequence
// and sets all of the Gradients accordingly.
func (s Sequence) PropagateLoss() float32 {
	var total float32
	for _, t := range s {
		total += SoftmaxLoss(t.Output, t.Target)
		t.Gradient = SoftmaxLossGrad(t.Output, t.Target)
	}
	return total / float32(len(s))
}

// Timestep represents a single timestep in a sequence.
type Timestep struct {
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

// TimestepSample points to a timestep in a sequence.
type TimestepSample struct {
	Sequence Sequence
	Index    int
}

// BranchFeature computes the value of the feature, which
// may be in the past.
func (t *TimestepSample) BranchFeature(b BranchFeature) bool {
	if b.StepsInPast > t.Index {
		return b.Feature == -1
	}
	if b.Feature == -1 {
		return false
	}
	ts := t.Sequence[t.Index-b.StepsInPast]
	return ts.Features[b.Feature]
}

// Timestep gets the corresponding Timestep.
func (t *TimestepSample) Timestep() *Timestep {
	return t.Sequence[t.Index]
}
