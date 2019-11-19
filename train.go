package seqtree

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

// BoundedStep computes a step size that ensures that the
// update KL divergence is less than maxKL and that the
// resulting loss improves.
func BoundedStep(timesteps []*TimestepSample, t *Tree, maxKL, maxStep float32) float32 {
	for i := 0; i < 64; i++ {
		var lock sync.Mutex
		var wg sync.WaitGroup
		var currentKL float32
		var currentDelta float32

		numProcs := runtime.GOMAXPROCS(0)
		for i := 0; i < numProcs; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				var klTotal, deltaTotal float32
				for j, ts := range timesteps {
					if j%numProcs != i {
						continue
					}
					leaf := t.Evaluate(ts)
					klTotal += SoftmaxLossKL(ts.Timestep().Output, leaf.OutputDelta, -maxStep)
					deltaTotal += SoftmaxLossDelta(ts.Timestep().Output, ts.Timestep().Target,
						leaf.OutputDelta, -maxStep)
				}
				lock.Lock()
				currentKL += klTotal
				currentDelta += deltaTotal
				lock.Unlock()
			}(i)
		}
		wg.Wait()
		currentKL /= float32(len(timesteps))
		if currentKL <= maxKL && currentDelta < 0 {
			return maxStep
		}
		maxStep *= 0.8
	}
	return 0
}

// OptimalStep performs a line search to find a step size
// that maximizes loss improvement.
func OptimalStep(timesteps []*TimestepSample, t *Tree, maxStep float32, iters int) float32 {
	minStep := float32(0.0)
	for i := 0; i < iters; i++ {
		midStep1 := minStep*0.75 + maxStep*0.25
		value1 := AvgLossDelta(timesteps, t, midStep1)
		midStep2 := minStep*0.25 + maxStep*0.75
		value2 := AvgLossDelta(timesteps, t, midStep2)
		if value2 < value1 {
			minStep = midStep1
		} else {
			maxStep = midStep2
		}
	}
	return (minStep + maxStep) / 2
}

// PropagateLosses computes the gradients for every
// sequence and sets the timesteps' Gradient fields.
func PropagateLosses(seqs []Sequence) {
	ch := make(chan Sequence, len(seqs))
	for _, seq := range seqs {
		ch <- seq
	}
	close(ch)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for seq := range ch {
				seq.PropagateLoss()
			}
		}()
	}

	wg.Wait()
}

// AvgLossDelta computes the average change in the loss
// after taking a step.
func AvgLossDelta(timesteps []*TimestepSample, t *Tree, currentStep float32) float32 {
	var lock sync.Mutex
	var currentDelta float32

	var wg sync.WaitGroup
	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			deltaTotal := newKahanSum(1)
			for j, ts := range timesteps {
				if j%numProcs != i {
					continue
				}
				leaf := t.Evaluate(ts)
				delta := SoftmaxLossDelta(ts.Timestep().Output, ts.Timestep().Target,
					leaf.OutputDelta, -currentStep)
				deltaTotal.Add([]float32{delta})
			}
			lock.Lock()
			currentDelta += deltaTotal.Sum()[0]
			lock.Unlock()
		}(i)
	}
	wg.Wait()
	return currentDelta / float32(len(timesteps))
}

// A Builder stores parameters for building new trees on
// top of a model.
type Builder struct {
	// Depth is the maximum depth of the resulting trees.
	Depth int

	// MinSplitSamples is the minimum number of samples
	// for splits to continue being made.
	// If a split results in a path with fewer than
	// MinSplitSamples, it will not be taken.
	MinSplitSamples int

	// MaxSplitSamples is the maximum number of samples to
	// use for calculating optimal splits. If there are
	// more samples than this, then a random subset of
	// samples are used.
	//
	// Setting this to zero has the special meaning of
	// using all of the samples for every split.
	MaxSplitSamples int

	// Horizons specifies the steps in the past to look at
	// features for splits.
	Horizons []int

	// ExtraFeatures, if non-zero, specifies the number of
	// features at the end of the feature list which
	// should be treated as second-class features and
	// should be used only sometimes rather than all the
	// time in splits.
	//
	// Specifying this can prevent slowdowns as more and
	// more features are added to the model.
	ExtraFeatures int
}

// Build builds a tree greedily using all of the provided
// samples. It is assumed that the samples already have a
// computed gradient.
func (b *Builder) Build(samples []*TimestepSample) *Tree {
	if len(samples) == 0 {
		panic("no data")
	}
	numFeatures := len(samples[0].Timestep().Features)
	return b.build(samples, b.Depth, numFeatures)
}

func (b *Builder) build(samples []*TimestepSample, depth, nextNewFeature int) *Tree {
	if depth == 0 || len(samples) <= b.MinSplitSamples {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: gradientMean(samples),
				Feature:     nextNewFeature,
			},
		}
	}

	splitSamples := samples
	if b.MaxSplitSamples != 0 && len(splitSamples) > b.MaxSplitSamples {
		splitSamples = make([]*TimestepSample, b.MaxSplitSamples)
		for i, j := range rand.Perm(len(splitSamples))[:b.MaxSplitSamples] {
			splitSamples[i] = samples[j]
		}
	}
	feature := optimalFeature(splitSamples, b.ExtraFeatures, b.Horizons)

	var falses, trues []*TimestepSample
	for _, t := range samples {
		if t.BranchFeature(feature) {
			trues = append(trues, t)
		} else {
			falses = append(falses, t)
		}
	}

	if len(trues) == 0 || len(falses) == 0 || len(trues) < b.MinSplitSamples ||
		len(falses) < b.MinSplitSamples {
		// No allowed split does any good.
		return b.build(samples, 0, nextNewFeature)
	}

	tree1 := b.build(falses, depth-1, nextNewFeature)
	tree2 := b.build(trues, depth-1, nextNewFeature+tree1.NumFeatures())

	return &Tree{
		Branch: &Branch{
			Feature:     feature,
			FalseBranch: tree1,
			TrueBranch:  tree2,
		},
	}
}

// optimalFeature finds the optimal feature to split on in
// order to separate the gradients of all the timesteps.
//
// This assumes that the timesteps all have a gradient
// set.
//
// The horizons argument specifies how many timesteps in
// the past we may look. A value of zero indicates that
// only the current timestep may be inspected.
func optimalFeature(samples []*TimestepSample, extraFeatures int, horizons []int) BranchFeature {
	if len(samples) == 0 {
		panic("no data")
	}
	numFeatures := len(samples[0].Timestep().Features)
	gradSum := gradientSum(samples)

	var resultLock sync.Mutex
	var bestFeature BranchFeature
	var bestQuality float32

	featureChan := make(chan BranchFeature, 10)
	wg := sync.WaitGroup{}
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range featureChan {
				quality := featureSplitQuality(samples, f, gradSum)
				resultLock.Lock()
				if quality >= bestQuality {
					bestFeature = f
					bestQuality = quality
				}
				resultLock.Unlock()
			}
		}()
	}

	for _, horizon := range horizons {
		for i := -1; i < numFeatures; i++ {
			if i >= numFeatures-extraFeatures {
				prob := 1 / math.Sqrt(float64(extraFeatures))
				if rand.Float64() > prob {
					continue
				}
			}
			featureChan <- BranchFeature{Feature: i, StepsInPast: horizon}
		}
	}
	close(featureChan)

	wg.Wait()

	return bestFeature
}

func featureSplitQuality(samples []*TimestepSample, f BranchFeature, sum []float32) float32 {
	falseCount := 0
	trueCount := 0
	featureValues := make([]bool, len(samples))

	for i, t := range samples {
		val := t.BranchFeature(f)
		featureValues[i] = val
		if val {
			trueCount++
		} else {
			falseCount++
		}
	}

	if falseCount == 0 || trueCount == 0 {
		return 0
	}

	minoritySum := make([]float32, len(sum))
	for i, val := range featureValues {
		if val == (trueCount < falseCount) {
			for j, x := range samples[i].Timestep().Gradient {
				minoritySum[j] += x
			}
		}
	}

	majoritySum := make([]float32, len(sum))
	for i, x := range sum {
		majoritySum[i] = x - minoritySum[i]
	}

	minorityCount := essentials.MinInt(falseCount, trueCount)
	majorityCount := essentials.MaxInt(falseCount, trueCount)

	return vectorNormSquared(minoritySum)/float32(minorityCount) +
		vectorNormSquared(majoritySum)/float32(majorityCount)
}

func vectorNormSquared(v []float32) float32 {
	var res float32
	for _, x := range v {
		res += x * x
	}
	return res
}

func gradientSum(ts []*TimestepSample) []float32 {
	sum := newKahanSum(len(ts[0].Timestep().Gradient))
	for _, t := range ts {
		sum.Add(t.Timestep().Gradient)
	}
	return sum.Sum()
}

func gradientMean(ts []*TimestepSample) []float32 {
	sum := gradientSum(ts)
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}
