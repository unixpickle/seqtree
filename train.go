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
// that minimizes the loss.
func OptimalStep(timesteps []*TimestepSample, t *Tree, maxStep float32, iters int) float32 {
	outputDeltas := make([][]float32, len(timesteps))
	for i, ts := range timesteps {
		outputDeltas[i] = t.Evaluate(ts).OutputDelta
	}

	// Golden section search:
	// https://en.wikipedia.org/wiki/Golden-section_search
	lossForStep := func(stepSize float32) float32 {
		var lock sync.Mutex
		var currentLoss float32

		var wg sync.WaitGroup
		numProcs := runtime.GOMAXPROCS(0)
		for i := 0; i < numProcs; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				total := newKahanSum(1)
				tmpAddition := []float32{0.0}
				tmpOutput := make([]float32, len(timesteps[0].Timestep().Output))
				for j := i; j < len(outputDeltas); j += numProcs {
					outputDelta := outputDeltas[j]
					ts := timesteps[j].Timestep()
					for i, x := range ts.Output {
						tmpOutput[i] = x - stepSize*outputDelta[i]
					}
					tmpAddition[0] = SoftmaxLoss(tmpOutput, ts.Target)
					total.Add(tmpAddition)
				}
				lock.Lock()
				currentLoss += total.Sum()[0]
				lock.Unlock()
			}(i)
		}
		wg.Wait()
		return currentLoss / float32(len(timesteps))
	}

	minStep := float32(0.0)
	var midValue1, midValue2 *float32

	for i := 0; i < iters; i++ {
		mid1 := maxStep - (maxStep-minStep)/math.Phi
		mid2 := minStep + (maxStep-minStep)/math.Phi
		if midValue1 == nil {
			x := lossForStep(mid1)
			midValue1 = &x
		}
		if midValue2 == nil {
			x := lossForStep(mid2)
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
	//
	// This is used as a lower bound. A few extra splits
	// may be tested, at negligible performance cost.
	CandidateSplits int

	// MaxUnion is the maximum number of features to
	// include in a union.
	// The special value 0 is treated as 1, indicating
	// that single features should be used.
	MaxUnion int

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

// build recursively creates a tree that splits up the
// samples in order to fit the functional gradient.
//
// The nextNewFeature argument specifies the current
// number of features, so that new leaves can be assigned
// unused feature numbers.
func (b *Builder) build(samples []*TimestepSample, depth, nextNewFeature int) *Tree {
	if depth == 0 || len(samples) <= b.MinSplitSamples {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: gradientMean(samples),
				Feature:     nextNewFeature,
			},
		}
	}
	return b.buildUnion(nil, samples, nil, depth, nextNewFeature)
}

// buildUnion is like build(), but it starts with a
// potentially non-empty set of split features to OR
// together.
//
// The falses argument specifies all of the samples which
// are negatively classified by the current union, while
// the trues argument specifies those samples which are
// positively classified by it.
//
// This function may modify the trues slice, but not the
// falses slice.
func (b *Builder) buildUnion(union BranchFeatureUnion, falses, trues []*TimestepSample, depth,
	nextNewFeature int) *Tree {
	if len(union) > 0 && len(union) >= b.MaxUnion {
		return b.buildSubtree(union, falses, trues, depth, nextNewFeature)
	}

	splitSamples := falses
	if b.MaxSplitSamples != 0 && len(splitSamples) > b.MaxSplitSamples {
		splitSamples = make([]*TimestepSample, b.MaxSplitSamples)
		for i, j := range rand.Perm(len(splitSamples))[:b.MaxSplitSamples] {
			splitSamples[i] = falses[j]
		}
	}
	sampleFrac := float32(float64(len(splitSamples)) / float64(len(falses)))
	features := b.sortFeatures(splitSamples, trues, sampleFrac)

	var bestFeature *BranchFeature
	if len(splitSamples) == len(falses) {
		// sortFeatures() gave an exact result.
		if len(features) > 0 {
			bestFeature = &features[0]
		}
	} else {
		bestFeature = b.optimalFeature(falses, trues, features)
	}

	if bestFeature == nil {
		return b.buildSubtree(union, falses, trues, depth, nextNewFeature)
	}

	var newFalses []*TimestepSample
	for _, sample := range falses {
		if sample.BranchFeature(*bestFeature) {
			trues = append(trues, sample)
		} else {
			newFalses = append(newFalses, sample)
		}
	}

	return b.buildUnion(append(union, *bestFeature), newFalses, trues, depth, nextNewFeature)
}

// buildSubtree creates the branches (or leaf) node for
// the given union and its resulting split.
func (b *Builder) buildSubtree(union BranchFeatureUnion, falses, trues []*TimestepSample,
	depth, nextNewFeature int) *Tree {
	if len(union) == 0 {
		return b.build(falses, 0, nextNewFeature)
	}
	tree1 := b.build(falses, depth-1, nextNewFeature)
	tree2 := b.build(trues, depth-1, nextNewFeature+tree1.NumFeatures())
	return &Tree{
		Branch: &Branch{
			Feature:     union,
			FalseBranch: tree1,
			TrueBranch:  tree2,
		},
	}
}

// optimalFeature finds the best feature from a set of
// ranked features.
//
// The features are added on to an existing union.
// The current split is indicated by falses and trues.
// It is assumed that the newly selected feature will act
// to move samples from falses into trues.
func (b *Builder) optimalFeature(falses, trues []*TimestepSample,
	features []BranchFeature) *BranchFeature {
	falseSum := gradientSum(falses, 0)
	trueSum := gradientSum(trues, len(falseSum))

	if b.CandidateSplits <= 1 {
		for _, feature := range features {
			quality := b.featureSplitQuality(falses, trues, falseSum, trueSum, feature, 1.0)
			if quality > 0 {
				return &feature
			}
		}
		return nil
	}

	var lock sync.Mutex
	var bestFeature BranchFeature
	var bestQuality float32
	var featuresTested int

	featureChan := make(chan BranchFeature, 0)
	var wg sync.WaitGroup

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for feature := range featureChan {
				quality := b.featureSplitQuality(falses, trues, falseSum, trueSum, feature, 1.0)
				if quality <= 0 {
					continue
				}
				lock.Lock()
				if featuresTested == 0 {
					bestFeature = feature
					bestQuality = quality
				} else if quality > bestQuality {
					bestFeature = feature
					bestQuality = quality
				}
				featuresTested++
				lock.Unlock()
			}
		}()
	}

	for _, feature := range features {
		featureChan <- feature
		lock.Lock()
		count := featuresTested
		lock.Unlock()
		// We will almost certainly check more than
		// b.CandidateSplits features, but that's
		// acceptable behavior.
		if count >= b.CandidateSplits {
			break
		}
	}
	close(featureChan)
	wg.Wait()

	if featuresTested == 0 {
		return nil
	}
	return &bestFeature
}

// sortFeatures finds features which produce reasonable
// splits and sorts them by quality.
//
// The falses and trues arguments represent the current
// split.
//
// This may be used with a subset of all of the
// falses, in order to improve performance.
// In this case, sampleFrac is less than 1.0 and indicates
// the fraction of the original falses slice that was
// passed.
// The trues argument is never a subset.
func (b *Builder) sortFeatures(falses, trues []*TimestepSample,
	sampleFrac float32) []BranchFeature {
	if len(falses) == 0 {
		panic("no data")
	}
	numFeatures := falses[0].Timestep().Features.Len()
	falseSum := gradientSum(falses, 0)
	trueSum := gradientSum(trues, len(falseSum))

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
				quality := b.featureSplitQuality(falses, trues, falseSum, trueSum, f, sampleFrac)
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

// featureSplitQuality evaluates a given split.
// The result is greater for better splits.
//
// The falseSum and trueSum arguments are the precomputed
// gradient sums for falses and trues, respectively.
// These sums are precomputed to improve performance.
//
// See sortFeatures() for details on sampleFrac.
func (b *Builder) featureSplitQuality(falses, trues []*TimestepSample, falseSum, trueSum []float32,
	f BranchFeature, sampleFrac float32) float32 {
	splitFalseCount := 0
	splitTrueCount := 0
	featureValues := make([]bool, len(falses))
	for i, t := range falses {
		val := t.BranchFeature(f)
		featureValues[i] = val
		if val {
			splitTrueCount++
		} else {
			splitFalseCount++
		}
	}

	approxTrues := float32(len(trues)) + float32(splitTrueCount)/sampleFrac
	approxFalses := float32(len(falses)-splitTrueCount) / sampleFrac

	if splitFalseCount == 0 || splitTrueCount == 0 ||
		int(approxTrues) < b.MinSplitSamples ||
		int(approxFalses) < b.MinSplitSamples {
		// The split is unlikely to be allowed.
		return 0
	}

	trueIsMinority := splitTrueCount < splitFalseCount

	minoritySum := make([]float32, len(falseSum))
	for i, val := range featureValues {
		if val == trueIsMinority {
			for j, x := range falses[i].Timestep().Gradient {
				minoritySum[j] += x
			}
		}
	}

	majoritySum := make([]float32, len(falseSum))
	for i, x := range falseSum {
		majoritySum[i] = x - minoritySum[i]
	}

	newTrueSum, newFalseSum := minoritySum, majoritySum
	if !trueIsMinority {
		newTrueSum, newFalseSum = majoritySum, minoritySum
	}

	for i, x := range trueSum {
		newTrueSum[i] += x * sampleFrac
	}
	effectiveTrueCount := float32(splitTrueCount) + float32(len(trues))*sampleFrac

	newQuality := vectorNormSquared(newFalseSum)/float32(splitFalseCount) +
		vectorNormSquared(newTrueSum)/effectiveTrueCount
	oldQuality := vectorNormSquared(falseSum)/float32(len(falses)) +
		vectorNormSquared(trueSum)*sampleFrac/float32(essentials.MaxInt(1, len(trues)))

	return newQuality - oldQuality
}

func vectorNormSquared(v []float32) float32 {
	var res float32
	for _, x := range v {
		res += x * x
	}
	return res
}

func gradientSum(ts []*TimestepSample, dim int) []float32 {
	if dim == 0 {
		dim = len(ts[0].Timestep().Gradient)
	}
	sum := newKahanSum(dim)
	for _, t := range ts {
		sum.Add(t.Timestep().Gradient)
	}
	return sum.Sum()
}

func gradientMean(ts []*TimestepSample) []float32 {
	sum := gradientSum(ts, 0)
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}
