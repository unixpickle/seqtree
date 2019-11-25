package seqtree

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

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

	// HigherOrder, if true, indicates that a higher-order
	// approximation of the loss function should be used
	// to choose optimal splits.
	//
	// This is currently only supported for binary
	// classification problems (i.e. problems with two
	// outputs).
	HigherOrder bool
}

// Build builds a tree greedily using all of the provided
// samples. It is assumed that the samples already have a
// computed gradient.
func (b *Builder) Build(samples []*TimestepSample) *Tree {
	if len(samples) == 0 {
		panic("no data")
	}
	ts := samples[0].Timestep()
	if b.HigherOrder && len(ts.Output) != 2 {
		panic("higher order optimization only supported for binary outputs")
	}
	numFeatures := ts.Features.Len()
	data := b.computeLossSamples(samples)
	return b.build(data, b.Depth, numFeatures)
}

// build recursively creates a tree that splits up the
// samples in order to fit the functional gradient.
//
// The nextNewFeature argument specifies the current
// number of features, so that new leaves can be assigned
// unused feature numbers.
func (b *Builder) build(samples []lossSample, depth, nextNewFeature int) *Tree {
	if depth == 0 || len(samples) <= b.MinSplitSamples {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: b.computeOutputDelta(samples),
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
func (b *Builder) buildUnion(union BranchFeatureUnion, falses, trues []lossSample, depth,
	nextNewFeature int) *Tree {
	if len(union) > 0 && len(union) >= b.MaxUnion {
		return b.buildSubtree(union, falses, trues, depth, nextNewFeature)
	}

	splitSamples := falses
	if b.MaxSplitSamples != 0 && len(splitSamples) > b.MaxSplitSamples {
		splitSamples = make([]lossSample, b.MaxSplitSamples)
		for i, j := range rand.Perm(len(splitSamples))[:b.MaxSplitSamples] {
			splitSamples[i] = falses[j]
		}
	}
	sampleFrac := float64(float64(len(splitSamples)) / float64(len(falses)))
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

	var newFalses []lossSample
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
func (b *Builder) buildSubtree(union BranchFeatureUnion, falses, trues []lossSample,
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
func (b *Builder) optimalFeature(falses, trues []lossSample, f []BranchFeature) *BranchFeature {
	sums := newLossSums(falses, trues)
	var bestFeature BranchFeature
	var bestQuality float64
	var featuresTested int

	for _, feature := range f {
		quality := b.featureSplitQuality(falses, trues, sums, feature, 1.0, true)
		if quality <= 0 {
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
// The falses and trues arguments represent the current
// split.
//
// This may be used with a subset of all of the
// falses, in order to improve performance.
// In this case, sampleFrac is less than 1.0 and indicates
// the fraction of the original falses slice that was
// passed.
// The trues argument is never a subset.
func (b *Builder) sortFeatures(falses, trues []lossSample, sampleFrac float64) []BranchFeature {
	if len(falses) == 0 {
		panic("no data")
	}
	numFeatures := falses[0].Timestep().Features.Len()
	sums := newLossSums(falses, trues)

	var resultLock sync.Mutex
	var features []BranchFeature
	var qualities []float64

	featureChan := make(chan BranchFeature, 10)
	wg := sync.WaitGroup{}
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range featureChan {
				quality := b.featureSplitQuality(falses, trues, sums, f, sampleFrac, false)
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
// See sortFeatures() for details on sampleFrac.
func (b *Builder) featureSplitQuality(falses, trues []lossSample, sums *lossSums, f BranchFeature,
	sampleFrac float64, parallel bool) float64 {
	featureValues, splitFalseCount, splitTrueCount := b.evaluateFeature(falses, f, parallel)

	approxTrues := float64(len(trues)) + float64(splitTrueCount)/sampleFrac
	approxFalses := float64(len(falses)-splitTrueCount) / sampleFrac

	if splitFalseCount == 0 || splitTrueCount == 0 ||
		int(approxTrues) < b.MinSplitSamples ||
		int(approxFalses) < b.MinSplitSamples {
		// The split is unlikely to be allowed.
		return 0
	}

	trueIsMinority := splitTrueCount < splitFalseCount
	minoritySum := b.minoritySum(falses, featureValues, trueIsMinority, parallel)
	majoritySum := make([]float64, len(sums.False))
	for i, x := range sums.False {
		majoritySum[i] = x - minoritySum[i]
	}

	newTrueSum, newFalseSum := minoritySum, majoritySum
	if !trueIsMinority {
		newTrueSum, newFalseSum = newFalseSum, newTrueSum
	}

	oldTrueSum := make([]float64, len(newTrueSum))
	for i, x := range sums.True {
		newTrueSum[i] += x * sampleFrac
		oldTrueSum[i] += x * sampleFrac
	}
	oldTrueCount := float64(len(trues)) * sampleFrac
	effectiveTrueCount := float64(splitTrueCount) + oldTrueCount

	newQuality := b.computeSplitQuality(newFalseSum, newTrueSum, float64(splitFalseCount),
		effectiveTrueCount)
	oldQuality := b.computeSplitQuality(sums.False, oldTrueSum, float64(len(falses)),
		oldTrueCount)

	return newQuality - oldQuality
}

func (b *Builder) evaluateFeature(samples []lossSample, f BranchFeature,
	parallel bool) (values []bool, falses, trues int) {
	byteIdx := f.Feature >> 3
	bitMask := byte(1) << uint8(f.Feature&7)
	if f.Feature == -1 {
		byteIdx = -1
	}

	values = make([]bool, len(samples))
	if !parallel {
		for i, s := range samples {
			val := s.branchFeatureFast(f.StepsInPast, byteIdx, bitMask)
			values[i] = val
			if val {
				trues++
			} else {
				falses++
			}
		}
		return
	}

	var wg sync.WaitGroup
	var lock sync.Mutex

	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var localTrues, localFalses int
			for j := i; j < len(samples); j += numProcs {
				val := samples[j].branchFeatureFast(f.StepsInPast, byteIdx, bitMask)
				values[j] = val
				if val {
					localTrues++
				} else {
					localFalses++
				}
			}
			lock.Lock()
			trues += localTrues
			falses += localFalses
			lock.Unlock()
		}(i)
	}

	wg.Wait()
	return
}

func (b *Builder) minoritySum(samples []lossSample, values []bool, trueIsMinority bool,
	parallel bool) []float64 {
	minoritySum := newKahanSum(len(samples[0].Vector))

	if !parallel {
		for i, val := range values {
			if val == trueIsMinority {
				minoritySum.Add(samples[i].Vector)
			}
		}
		return minoritySum.Sum()
	}

	var wg sync.WaitGroup
	var lock sync.Mutex

	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			localSum := newKahanSum(len(samples[0].Vector))
			for j := i; j < len(samples); j += numProcs {
				if values[j] == trueIsMinority {
					localSum.Add(samples[j].Vector)
				}
			}
			lock.Lock()
			minoritySum.Add(localSum.Sum())
			lock.Unlock()
		}(i)
	}

	wg.Wait()
	return minoritySum.Sum()
}

func (b *Builder) computeSplitQuality(falses, trues []float64,
	falseCount, trueCount float64) float64 {
	if b.HigherOrder {
		poly1 := polynomial(falses)
		min1 := b.minimizePolynomial(poly1)
		poly2 := polynomial(trues)
		min2 := b.minimizePolynomial(poly2)
		return -(poly1.Evaluate(min1) + poly2.Evaluate(min2))
	}
	if trueCount == 0 {
		trueCount = 1
	} else if falseCount == 0 {
		falseCount = 1
	}
	return vectorNormSquared(falses)/falseCount + vectorNormSquared(trues)/trueCount
}

func (b *Builder) computeLossSamples(samples []*TimestepSample) []lossSample {
	res := make([]lossSample, len(samples))

	numProcs := runtime.GOMAXPROCS(0)
	wg := sync.WaitGroup{}
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := i; j < len(samples); j += numProcs {
				sample := samples[j]
				ts := sample.Timestep()
				res[j].TimestepSample = *sample
				if b.HigherOrder {
					poly := newPolynomialLogSigmoid(ts.Output[0] - ts.Output[1])
					poly1 := newPolynomialLogSigmoid(ts.Output[1] - ts.Output[0])
					for i, x := range poly {
						poly[i] = -x * ts.Target[0]
						if i%2 == 1 {
							poly[i] += poly1[i] * ts.Target[1]
						} else {
							poly[i] -= poly1[i] * ts.Target[1]
						}
					}

					// Getting rid of the constant term improves
					// numerical accuracy without changing the
					// objective
					poly[0] = 0

					res[j].Vector = poly
				} else {
					grad := SoftmaxLossGrad(ts.Output, ts.Target)
					res[j].Vector = grad
				}
			}
		}(i)
	}
	wg.Wait()

	return res
}

func (b *Builder) computeOutputDelta(samples []lossSample) []float64 {
	sum := newKahanSum(len(samples[0].Vector))
	for _, s := range samples {
		sum.Add(s.Vector)
	}
	res := sum.Sum()

	if b.HigherOrder {
		x := b.minimizePolynomial(polynomial(res))
		return []float64{-x, x}
	}

	for i, x := range res {
		res[i] = x / float64(len(samples))
	}
	return res
}

func (b *Builder) minimizePolynomial(p polynomial) float64 {
	return minimizeUnary(-1, 1, 60, p.Evaluate)
}

type lossSample struct {
	TimestepSample

	// Vector is some linear representation of the loss
	// function for this sample.
	// It may be a gradient, or a set of polynomial
	// coefficients.
	Vector []float64
}

func (l *lossSample) branchFeatureFast(stepsInPast, byteIdx int, bitMask byte) bool {
	if stepsInPast > l.Index {
		return byteIdx == -1
	} else if byteIdx == -1 {
		return false
	}
	ts := l.Sequence[l.Index-stepsInPast]
	return ts.Features.bytes[byteIdx]&bitMask != 0
}

type lossSums struct {
	False []float64
	True  []float64
}

func newLossSums(falses, trues []lossSample) *lossSums {
	falseSum := newKahanSum(len(falses[0].Vector))
	trueSum := newKahanSum(len(falseSum.Sum()))
	for _, s := range falses {
		falseSum.Add(s.Vector)
	}
	for _, s := range trues {
		trueSum.Add(s.Vector)
	}
	return &lossSums{False: falseSum.Sum(), True: trueSum.Sum()}
}
