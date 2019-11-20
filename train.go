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
	// Golden section search:
	// https://en.wikipedia.org/wiki/Golden-section_search

	minStep := float32(0.0)
	var midValue1, midValue2 *float32

	for i := 0; i < iters; i++ {
		mid1 := maxStep - (maxStep-minStep)/math.Phi
		mid2 := minStep + (maxStep-minStep)/math.Phi
		if midValue1 == nil {
			x := AvgLossDelta(timesteps, t, mid1)
			midValue1 = &x
		}
		if midValue2 == nil {
			x := AvgLossDelta(timesteps, t, mid2)
			midValue2 = &x
		}

		if *midValue2 < *midValue1 {
			minStep = mid1
			midValue1 = midValue2
			midValue2 = nil
		} else {
			maxStep = mid2
			midValue2 = midValue1
			midValue1 = nil
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
	// use for sorting features for splits. If there are
	// more samples than this, then a random subset of
	// samples are used.
	//
	// Setting this to zero has the special meaning of
	// using all of the samples for every split.
	MaxSplitSamples int

	// CandidateSplits specifies the number of top splits
	// to evaluate on the full set of samples.
	// This can be used with MaxSplitSamples to find good
	// splits using a subset of the data, and then pick
	// the best of these using all of the data.
	//
	// If zero, the top usable split is selected.
	CandidateSplits int

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
	numFeatures := samples[0].Timestep().Features.Len()
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
	features := b.sortFeatures(splitSamples, float32(len(splitSamples))/float32(len(samples)))

	var bestFeature *BranchFeature
	if len(splitSamples) == len(samples) {
		if len(features) > 0 {
			bestFeature = &features[0]
		}
	} else {
		bestFeature = b.optimalFeature(samples, features)
	}

	if bestFeature == nil {
		return b.build(samples, 0, nextNewFeature)
	}

	var falses, trues []*TimestepSample
	for _, t := range samples {
		if t.BranchFeature(*bestFeature) {
			trues = append(trues, t)
		} else {
			falses = append(falses, t)
		}
	}

	tree1 := b.build(falses, depth-1, nextNewFeature)
	tree2 := b.build(trues, depth-1, nextNewFeature+tree1.NumFeatures())
	return &Tree{
		Branch: &Branch{
			Feature:     *bestFeature,
			FalseBranch: tree1,
			TrueBranch:  tree2,
		},
	}
}

// optimalFeature finds the best feature from a set of
// ranked features.
func (b *Builder) optimalFeature(samples []*TimestepSample,
	features []BranchFeature) *BranchFeature {
	sum := gradientSum(samples)

	var bestFeature BranchFeature
	var bestQuality float32
	var featuresTested int

	for _, feature := range features {
		quality := b.featureSplitQuality(samples, feature, sum, 1.0)
		if quality == 0 {
			continue
		}

		if featuresTested == 0 {
			bestFeature = feature
			bestQuality = quality
		} else if quality > bestQuality {
			bestFeature = feature
			bestQuality = quality
		}

		featuresTested++
		if featuresTested >= b.CandidateSplits {
			break
		}
	}
	if featuresTested == 0 {
		return nil
	}
	return &bestFeature
}

// sortFeatures finds features which produce reasonable
// splits and sorts them by quality.
//
// This assumes that the timesteps all have a gradient
// set.
func (b *Builder) sortFeatures(samples []*TimestepSample, sampleFrac float32) []BranchFeature {
	if len(samples) == 0 {
		panic("no data")
	}
	numFeatures := samples[0].Timestep().Features.Len()
	gradSum := gradientSum(samples)

	var resultLock sync.Mutex
	var features []BranchFeature
	var qualities []float32

	featureChan := make(chan BranchFeature, 10)
	wg := sync.WaitGroup{}
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range featureChan {
				quality := b.featureSplitQuality(samples, f, gradSum, sampleFrac)
				if quality > 0 {
					resultLock.Lock()
					features = append(features, f)
					qualities = append(qualities, quality)
					resultLock.Unlock()
				}
			}
		}()
	}

	for _, horizon := range b.Horizons {
		for i := -1; i < numFeatures; i++ {
			if i >= numFeatures-b.ExtraFeatures {
				prob := 1 / math.Sqrt(float64(b.ExtraFeatures))
				if rand.Float64() > prob {
					continue
				}
			}
			featureChan <- BranchFeature{Feature: i, StepsInPast: horizon}
		}
	}
	close(featureChan)
	wg.Wait()

	essentials.VoodooSort(qualities, func(i, j int) bool {
		return qualities[i] > qualities[j]
	}, features)

	return features
}

func (b *Builder) featureSplitQuality(samples []*TimestepSample, f BranchFeature,
	sum []float32, sampleFrac float32) float32 {
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

	minSamples := int(sampleFrac * float32(b.MinSplitSamples))
	if falseCount == 0 || trueCount == 0 || trueCount < minSamples ||
		falseCount < minSamples {
		// The split is unlikely to be allowed.
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
