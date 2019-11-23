package seqtree

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

// OptimalStep performs a line search to find a step size
// that minimizes the loss.
func OptimalStep(timesteps []*TimestepSample, t *Tree, maxStep float32, iters int) float32 {
	outputDeltas := make([][]float32, len(timesteps))
	for i, ts := range timesteps {
		outputDeltas[i] = t.Evaluate(ts).OutputDelta
	}

	return minimizeUnary(0, maxStep, iters, func(stepSize float32) float32 {
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
		return currentLoss
	})
}

// ScaleOptimalStep scales the leaves of t individually to
// minimize the loss when a step of size 1 is taken.
//
// The maxStep argument is the maximum scaling for a leaf.
// The minLeafSamples argument is the minimum number of
// representative samples a leaf must have in order to be
// scaled.
func ScaleOptimalStep(timesteps []*TimestepSample, t *Tree, maxStep float32,
	minLeafSamples, iters int) {
	leafToSample := map[*Leaf][]*Timestep{}
	for _, ts := range timesteps {
		leaf := t.Evaluate(ts)
		leafToSample[leaf] = append(leafToSample[leaf], ts.Timestep())
	}
	for leaf, samples := range leafToSample {
		if len(samples) < minLeafSamples || len(samples) == 0 {
			continue
		}
		scale := minimizeUnary(0, maxStep, iters, func(stepSize float32) float32 {
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
					tmpOutput := make([]float32, len(samples[0].Output))
					for j := i; j < len(samples); j += numProcs {
						sample := samples[j]
						for k, x := range sample.Output {
							tmpOutput[k] = x - stepSize*leaf.OutputDelta[k]
						}
						tmpAddition[0] = SoftmaxLoss(tmpOutput, sample.Target)
						total.Add(tmpAddition)
					}
					lock.Lock()
					currentLoss += total.Sum()[0]
					lock.Unlock()
				}(i)
			}
			wg.Wait()
			return currentLoss
		})
		for i, x := range leaf.OutputDelta {
			leaf.OutputDelta[i] = x * scale
		}
	}
}

// PropagateLosses computes the gradients for every
// sequence and sets the timesteps' Gradient fields.
func PropagateLosses(seqs []Sequence) {
	propagateLosses(seqs, false, false)
}

// PropagateLossesNatural is like PropagateLosses, except
// that it uses the natural gradient.
func PropagateLossesNatural(seqs []Sequence) {
	propagateLosses(seqs, true, false)
}

// PropagateHessians computes the hessians for every
// sequence and sets the timesteps' Hessian fields.
func PropagateHessians(seqs []Sequence) {
	propagateLosses(seqs, false, true)
}

func propagateLosses(seqs []Sequence, natural, hessian bool) {
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
				if natural {
					seq.PropagateLossNatural()
				} else if hessian {
					seq.PropagateHessian()
				} else {
					seq.PropagateLoss()
				}
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
	sums := newGradientSums(falses, trues)

	if b.CandidateSplits <= 1 {
		for _, feature := range features {
			quality := b.featureSplitQuality(falses, trues, sums, feature, 1.0)
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
				quality := b.featureSplitQuality(falses, trues, sums, feature, 1.0)
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
	sums := newGradientSums(falses, trues)

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
// The falseSum and trueSum arguments are the precomputed
// gradient sums for falses and trues, respectively.
// These sums are precomputed to improve performance.
//
// See sortFeatures() for details on sampleFrac.
func (b *Builder) featureSplitQuality(falses, trues []*TimestepSample, sums *gradientSums,
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

	minoritySum := newKahanSum(len(sums.False))
	minorityHessianSum := newKahanSum(len(sums.FalseHessian))
	for i, val := range featureValues {
		if val == trueIsMinority {
			minoritySum.Add(falses[i].Timestep().Gradient)
			minorityHessianSum.Add(falses[i].Timestep().Hessian.Data)
		}
	}

	majoritySum := make([]float32, len(sums.False))
	majorityHessianSum := make([]float32, len(sums.FalseHessian))
	for i, x := range sums.False {
		majoritySum[i] = x - minoritySum.Sum()[i]
	}
	for i, x := range sums.FalseHessian {
		majorityHessianSum[i] = x - minorityHessianSum.Sum()[i]
	}

	newTrueSum, newFalseSum := minoritySum.Sum(), majoritySum
	newTrueHessianSum, newFalseHessianSum := minorityHessianSum.Sum(), majorityHessianSum
	if !trueIsMinority {
		newTrueSum, newFalseSum = newFalseSum, newTrueSum
		newTrueHessianSum, newFalseHessianSum = newFalseHessianSum, newTrueHessianSum
	}

	for i, x := range sums.True {
		newTrueSum[i] += x * sampleFrac
	}
	for i, x := range sums.TrueHessian {
		newTrueHessianSum[i] += x * sampleFrac
	}

	return hessianSplitQuality(newTrueSum, newTrueHessianSum) +
		hessianSplitQuality(newFalseSum, newFalseHessianSum)
}

func hessianSplitQuality(grad []float32, hessian []float32) float32 {
	h := &SquareMatrix{Dim: len(grad), Data: hessian}
	// solution := h.Inverse().VectorProduct(grad)
	// for i := range solution {
	// 	solution[i] *= -1
	// }
	solution := []float32{-grad[0] / hessian[0], 0}
	product := h.VectorProduct(solution)
	var result float32
	for i, g := range grad {
		result += g*solution[i] + 0.5*product[i]*solution[i]
	}
	return -result
}

type gradientSums struct {
	False []float32
	True  []float32

	FalseHessian []float32
	TrueHessian  []float32
}

func newGradientSums(falses, trues []*TimestepSample) *gradientSums {
	falseSum := gradientSum(falses, 0)
	trueSum := gradientSum(trues, len(falseSum))
	falseHessianSum := hessianSum(falses, 0)
	trueHessianSum := hessianSum(trues, len(falseHessianSum))
	return &gradientSums{
		False:        falseSum,
		True:         trueSum,
		FalseHessian: falseHessianSum,
		TrueHessian:  trueHessianSum,
	}
}

func hessianSum(ts []*TimestepSample, dim int) []float32 {
	if dim == 0 {
		dim = len(ts[0].Timestep().Hessian.Data)
	}
	sum := newKahanSum(dim)
	for _, t := range ts {
		sum.Add(t.Timestep().Hessian.Data)
	}
	return sum.Sum()
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
