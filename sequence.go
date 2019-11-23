package seqtree

// MakeOneHotSequence creates a sequence for a slice of
// one-hot values.
// The inputs at each timestep are the previous values,
// and the outputs are the current values.
// The final output is 0, and the initial input has no
// features set.
func MakeOneHotSequence(seq []int, outputSize, numFeatures int) Sequence {
	ts := &Timestep{
		Features: NewBitmap(numFeatures),
		Output:   make([]float32, outputSize),
		Target:   make([]float32, outputSize),
	}
	res := Sequence{}
	for _, x := range seq {
		ts.Target[x] = 1.0
		res = append(res, ts)
		ts = &Timestep{
			Features: NewBitmap(numFeatures),
			Output:   make([]float32, outputSize),
			Target:   make([]float32, outputSize),
		}
		ts.Features.Set(x, true)
	}
	ts.Target[0] = 1.0
	return append(res, ts)
}

type Sequence []*Timestep

// MeanLoss computes the mean loss for the sequence.
func (s Sequence) MeanLoss() float32 {
	var total float32
	for _, t := range s {
		total += SoftmaxLoss(t.Output, t.Target)
	}
	return total / float32(len(s))
}

// Timestep represents a single timestep in a sequence.
type Timestep struct {
	// Features stores the current feature bitmap.
	Features *Bitmap

	// Output is the current prediction parameter vector
	// for this timestamp.
	Output []float32

	// Target is a vector of output probabilities
	// representing the ground truth label.
	Target []float32
}

// TimestepSample points to a timestep in a sequence.
type TimestepSample struct {
	Sequence Sequence
	Index    int
}

// TimestepSamples gets all of the timestep samples from
// all of the sequences.
func TimestepSamples(seqs []Sequence) []*TimestepSample {
	var res []*TimestepSample
	for _, seq := range seqs {
		for i := range seq {
			res = append(res, &TimestepSample{Sequence: seq, Index: i})
		}
	}
	return res
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
	return ts.Features.Get(b.Feature)
}

// Timestep gets the corresponding Timestep.
func (t *TimestepSample) Timestep() *Timestep {
	return t.Sequence[t.Index]
}

// A Bitmap is effectively an array of booleans.
type Bitmap struct {
	numBits int
	bytes   []uint8
}

// NewBitmap creates a bitmap of all zeros.
func NewBitmap(numBits int) *Bitmap {
	numBytes := numBits / 8
	if numBits%8 != 0 {
		numBytes++
	}
	return &Bitmap{numBits: numBits, bytes: make([]uint8, numBytes)}
}

// Len gets the number of bits.
func (b *Bitmap) Len() int {
	return b.numBits
}

// Get gets the bit at index i.
func (b *Bitmap) Get(i int) bool {
	if i < 0 || i >= b.numBits {
		panic("index out of range")
	}
	return b.bytes[i>>3]&(1<<uint(i&7)) != 0
}

// Set sets the bit at index i.
func (b *Bitmap) Set(i int, v bool) {
	if i < 0 || i >= b.numBits {
		panic("index out of range")
	}
	if v {
		b.bytes[i>>3] |= 1 << uint(i&7)
	} else {
		b.bytes[i>>3] &= ^(1 << uint(i&7))
	}
}
