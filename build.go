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

	// CandidatePruneSamples is the number of samples to
	// test each candidate split on before testing it on
	// the full dataset.
	CandidatePruneSamples int

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
	// In the general case, this uses a second order
	// approximation of the loss function.
	// In the case of binary outputs, this uses a higher
	// order polynomial.
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

	splitSamples, sampleFrac := subsampleLimit(falses, b.MaxSplitSamples)
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

	pruneFalses, pruneFrac := subsampleLimit(falses, b.CandidatePruneSamples)
	pruneSums := newLossSums(pruneFalses, trues)

	var lock sync.Mutex
	var bestFeature BranchFeature
	var bestQuality float32
	var successfulFeatures int
	var currentFeature int

	getNext := func() *BranchFeature {
		lock.Lock()
		defer lock.Unlock()
		if successfulFeatures >= essentials.MaxInt(1, b.CandidateSplits) {
			return nil
		} else if currentFeature == len(f) {
			return nil
		}
		res := &f[currentFeature]
		currentFeature++
		return res
	}

	putResult := func(f BranchFeature, quality float32) {
		lock.Lock()
		defer lock.Unlock()
		if quality <= 0 {
			return
		}
		if quality > bestQuality || successfulFeatures == 0 {
			bestQuality = quality
			bestFeature = f
		}
		successfulFeatures++
	}

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				fPtr := getNext()
				if fPtr == nil {
					return
				}
				feature := *fPtr
				quality := b.featureSplitQuality(pruneFalses, trues, pruneSums, feature, pruneFrac)
				if quality > 0 && len(pruneFalses) != len(falses) {
					quality = b.featureSplitQuality(falses, trues, sums, feature, 1.0)
				}
				putResult(feature, quality)
			}
		}()
	}
	wg.Wait()

	if successfulFeatures == 0 {
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
func (b *Builder) sortFeatures(falses, trues []lossSample, sampleFrac float32) []BranchFeature {
	if len(falses) == 0 {
		panic("no data")
	}
	numFeatures := falses[0].Timestep().Features.Len()
	sums := newLossSums(falses, trues)

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
				quality := b.featureSplitQuality(falses, trues, sums, f, sampleFrac)
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
	sampleFrac float32) float32 {
	featureValues, splitFalseCount, splitTrueCount := b.evaluateFeature(falses, f)

	approxTrues := float32(len(trues)) + float32(splitTrueCount)/sampleFrac
	approxFalses := float32(len(falses)-splitTrueCount) / sampleFrac

	if splitFalseCount == 0 || splitTrueCount == 0 ||
		int(approxTrues) < b.MinSplitSamples ||
		int(approxFalses) < b.MinSplitSamples {
		// The split is unlikely to be allowed.
		return 0
	}

	trueIsMinority := splitTrueCount < splitFalseCount
	minoritySum := b.minoritySum(falses, featureValues, trueIsMinority)
	majoritySum := make([]float32, len(sums.False))
	for i, x := range sums.False {
		majoritySum[i] = x - minoritySum[i]
	}

	newTrueSum, newFalseSum := minoritySum, majoritySum
	if !trueIsMinority {
		newTrueSum, newFalseSum = newFalseSum, newTrueSum
	}

	oldTrueSum := make([]float32, len(newTrueSum))
	for i, x := range sums.True {
		newTrueSum[i] += x * sampleFrac
		oldTrueSum[i] += x * sampleFrac
	}
	oldTrueCount := float32(len(trues)) * sampleFrac
	effectiveTrueCount := float32(splitTrueCount) + oldTrueCount

	numOutputs := len(falses[0].Timestep().Output)
	newQuality := b.computeSplitQuality(newFalseSum, newTrueSum, float32(splitFalseCount),
		effectiveTrueCount, numOutputs)
	oldQuality := b.computeSplitQuality(sums.False, oldTrueSum, float32(len(falses)),
		oldTrueCount, numOutputs)

	// Avoid numerically insignificant deltas.
	minDelta := math.Abs(math.Min(float64(newQuality), float64(oldQuality))) * 1e-6
	if math.Abs(float64(newQuality-oldQuality)) < minDelta {
		return 0
	}

	return newQuality - oldQuality
}

func (b *Builder) evaluateFeature(samples []lossSample, f BranchFeature) (values []bool,
	falses, trues int) {
	byteIdx := f.Feature >> 3
	bitMask := byte(1) << uint8(f.Feature&7)
	if f.Feature == -1 {
		byteIdx = -1
	}

	values = make([]bool, len(samples))
	for i, s := range samples {
		val := s.BranchFeatureFast(f.StepsInPast, byteIdx, bitMask)
		values[i] = val
		if val {
			trues++
		} else {
			falses++
		}
	}
	return
}

func (b *Builder) minoritySum(samples []lossSample, values []bool, trueIsMinority bool) []float32 {
	minoritySum := newKahanSum(len(samples[0].Vector))
	for i, val := range values {
		if val == trueIsMinority {
			minoritySum.Add(samples[i].Vector)
		}
	}
	return minoritySum.Sum()
}

func (b *Builder) computeSplitQuality(falses, trues []float32,
	falseCount, trueCount float32, numOutputs int) float32 {
	if b.HigherOrder {
		if numOutputs == 2 {
			poly1 := polynomial(falses)
			min1 := b.minimizePolynomial(poly1)
			poly2 := polynomial(trues)
			min2 := b.minimizePolynomial(poly2)
			return -(poly1.Evaluate(min1) + poly2.Evaluate(min2))
		} else {
			_, min1 := b.minimizeSecondOrder(falses, numOutputs)
			_, min2 := b.minimizeSecondOrder(trues, numOutputs)
			return -(min1 + min2)
		}
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
					numOutputs := len(ts.Output)
					if numOutputs == 2 {
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
						hess := newHessianMatrixSoftmax(ts.Output, ts.Target)
						res[j].Vector = append(grad, hess.Values...)
					}
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

func (b *Builder) computeOutputDelta(samples []lossSample) []float32 {
	sum := newKahanSum(len(samples[0].Vector))
	for _, s := range samples {
		sum.Add(s.Vector)
	}
	res := sum.Sum()

	if b.HigherOrder {
		numOutputs := len(samples[0].Timestep().Output)
		if numOutputs == 2 {
			x := b.minimizePolynomial(polynomial(res))
			return []float32{-x, x}
		} else {
			solution, _ := b.minimizeSecondOrder(res, numOutputs)
			for i := range solution {
				solution[i] *= -1
			}
			return solution
		}
	}

	for i, x := range res {
		res[i] = x / float32(len(samples))
	}
	return res
}

func (b *Builder) minimizePolynomial(p polynomial) float32 {
	return minimizeUnary(-1, 1, 30, p.Evaluate)
}

func (b *Builder) minimizeSecondOrder(data []float32, numOutputs int) ([]float32, float32) {
	grad := data[:numOutputs]
	hessian := &hessianMatrix{
		Dim:    numOutputs,
		Values: data[numOutputs:],
	}
	negGrad := make([]float32, len(grad))
	for i, x := range grad {
		negGrad[i] = -x
	}
	solution := hessian.ApplyInverse(negGrad)
	value := vectorDot(grad, solution) + 0.5*vectorDot(solution, hessian.Apply(solution))
	return solution, value
}

type lossSample struct {
	TimestepSample

	// Vector is some linear representation of the loss
	// function for this sample.
	// It may be a gradient, or a set of polynomial
	// coefficients, or a combination of a gradient and a
	// hessian matrix.
	Vector []float32
}

func (l *lossSample) BranchFeatureFast(stepsInPast, byteIdx int, bitMask byte) bool {
	if stepsInPast > l.Index {
		return byteIdx == -1
	} else if byteIdx == -1 {
		return false
	}
	ts := l.Sequence[l.Index-stepsInPast]
	return ts.Features.bytes[byteIdx]&bitMask != 0
}

type lossSums struct {
	False []float32
	True  []float32
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

func subsampleLimit(samples []lossSample, max int) ([]lossSample, float32) {
	splitSamples := samples
	if max != 0 && len(splitSamples) > max {
		splitSamples = make([]lossSample, max)
		for i, j := range rand.Perm(len(samples))[:max] {
			splitSamples[i] = samples[j]
		}
	}
	sampleFrac := float32(float64(len(splitSamples)) / float64(len(samples)))
	return splitSamples, sampleFrac
}
