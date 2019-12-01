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
	// Heuristic is used to find good splits and compute
	// output nodes.
	Heuristic Heuristic

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
}

// Build builds a tree greedily using all of the provided
// samples. It is assumed that the samples already have a
// computed gradient.
func (b *Builder) Build(samples []*TimestepSample) *Tree {
	if len(samples) == 0 {
		panic("no data")
	}
	if b.Heuristic == nil {
		panic("no heuristic was specified")
	}
	data := newVecSamples(b.Heuristic, samples)
	return b.build(data, b.Depth)
}

// build recursively creates a tree that splits up the
// samples in order to fit the functional gradient.
func (b *Builder) build(samples []vecSample, depth int) *Tree {
	if depth == 0 || len(samples) <= b.MinSplitSamples {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: vecSamplesOutputDelta(b.Heuristic, samples),
			},
		}
	}
	return b.buildUnion(nil, samples, nil, depth)
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
func (b *Builder) buildUnion(union BranchFeatureUnion, falses, trues []vecSample,
	depth int) *Tree {
	if len(union) > 0 && len(union) >= b.MaxUnion {
		return b.buildSubtree(union, falses, trues, depth)
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
		return b.buildSubtree(union, falses, trues, depth)
	}

	var newFalses []vecSample
	for _, sample := range falses {
		if sample.BranchFeature(*bestFeature) {
			trues = append(trues, sample)
		} else {
			newFalses = append(newFalses, sample)
		}
	}

	return b.buildUnion(append(union, *bestFeature), newFalses, trues, depth)
}

// buildSubtree creates the branches (or leaf) node for
// the given union and its resulting split.
func (b *Builder) buildSubtree(union BranchFeatureUnion, falses, trues []vecSample,
	depth int) *Tree {
	if len(union) == 0 {
		return b.build(falses, 0)
	}
	tree1 := b.build(falses, depth-1)
	tree2 := b.build(trues, depth-1)
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
func (b *Builder) optimalFeature(falses, trues []vecSample, f []BranchFeature) *BranchFeature {
	sums := newLossSums(falses, trues)

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
				quality := b.featureSplitQuality(falses, trues, sums, feature, 1.0)
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
func (b *Builder) sortFeatures(falses, trues []vecSample, sampleFrac float32) []BranchFeature {
	if len(falses) == 0 {
		panic("no data")
	}

	totalSum := newLossSums(falses, trues)
	for i, x := range totalSum.True {
		totalSum.True[i] = x * sampleFrac
	}
	baseQuality := b.Heuristic.Quality(totalSum.False) + b.Heuristic.Quality(totalSum.True)

	counts := b.countFeatureOccurrences(falses)
	usable, trueIsMinority := b.filterFeatures(counts, len(falses), len(trues), sampleFrac)
	sums := b.sumMinorities(falses, counts, usable, trueIsMinority)

	var lock sync.Mutex
	var horizonIdx, featureIdx int
	getNext := func() (int, int, bool) {
		lock.Lock()
		defer lock.Unlock()
		if horizonIdx >= len(sums) {
			return 0, 0, false
		}
		for featureIdx >= len(sums[horizonIdx]) {
			horizonIdx++
			featureIdx = 0
			if horizonIdx == len(sums) {
				return 0, 0, false
			}
		}
		f := featureIdx
		featureIdx++
		return horizonIdx, f, true
	}

	var resultLock sync.Mutex
	var resultingFeatures []BranchFeature
	var resultingQualities []float32

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i, j, more := getNext()
				if !more {
					return
				}
				horizon := b.Horizons[i]
				feature := usable[i][j]
				sum := sums[i][j].Sum()
				tIsMin := trueIsMinority[i][j]

				majoritySum := make([]float32, len(sum))
				for k, x := range totalSum.False {
					majoritySum[k] = x - sum[k]
				}
				trueSum, falseSum := sum, majoritySum
				if !tIsMin {
					trueSum, falseSum = falseSum, trueSum
				}
				for k, x := range totalSum.True {
					trueSum[k] += x
				}
				quality := b.Heuristic.Quality(trueSum) + b.Heuristic.Quality(falseSum) - baseQuality
				if quality > 1e-6*baseQuality {
					resultLock.Lock()
					resultingQualities = append(resultingQualities, quality)
					resultingFeatures = append(resultingFeatures, BranchFeature{
						Feature:     feature,
						StepsInPast: horizon,
					})
					resultLock.Unlock()
				}
			}
		}()
	}
	wg.Wait()

	essentials.VoodooSort(resultingQualities, func(i, j int) bool {
		return resultingQualities[i] > resultingQualities[j]
	}, resultingFeatures)

	return resultingFeatures
}

func (b *Builder) countFeatureOccurrences(samples []vecSample) [][]int {
	numFeatures := samples[0].Timestep().Features.Len() + 1
	makeCounts := func() [][]int {
		res := make([][]int, len(b.Horizons))
		for i := range res {
			res[i] = make([]int, numFeatures)
		}
		return res
	}

	var lock sync.Mutex
	var wg sync.WaitGroup
	sum := makeCounts()
	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			localCounts := makeCounts()
			for j := i; j < len(samples); j += numProcs {
				sample := samples[j]
				for k, counts := range localCounts {
					horizon := b.Horizons[k]
					if horizon > sample.Index {
						counts[0]++
					} else {
						ts := sample.Sequence[sample.Index-horizon]
						for i := 1; i < numFeatures; i++ {
							if ts.Features.Get(i - 1) {
								counts[i]++
							}
						}
					}
				}
			}
			lock.Lock()
			for i, counts := range localCounts {
				for j, c := range counts {
					sum[i][j] += c
				}
			}
			lock.Unlock()
		}(i)
	}
	wg.Wait()

	return sum
}

func (b *Builder) filterFeatures(counts [][]int, falseCount, trueCount int,
	sampleFrac float32) ([][]int, [][]bool) {
	var features [][]int
	var trueIsMinority [][]bool
	for _, horizonCounts := range counts {
		var horizonFeatures []int
		var horizonTrueIsMinority []bool
		for i, n := range horizonCounts {
			splitTrueCount := n
			splitFalseCount := falseCount - splitTrueCount

			approxTrues := float32(trueCount) + float32(splitTrueCount)/sampleFrac
			approxFalses := float32(splitFalseCount) / sampleFrac

			if splitFalseCount == 0 || splitTrueCount == 0 ||
				int(approxTrues) < b.MinSplitSamples ||
				int(approxFalses) < b.MinSplitSamples {
				// The split is unlikely to be allowed.
				continue
			}

			horizonFeatures = append(horizonFeatures, i-1)
			horizonTrueIsMinority = append(horizonTrueIsMinority, splitTrueCount < splitFalseCount)
		}
		features = append(features, horizonFeatures)
		trueIsMinority = append(trueIsMinority, horizonTrueIsMinority)
	}
	return features, trueIsMinority
}

func (b *Builder) sumMinorities(samples []vecSample, counts, features [][]int,
	trueIsMinority [][]bool) [][]kahanSum {
	vecSize := len(samples[0].Vector)
	makeSums := func() [][]kahanSum {
		bufSize := 0
		for _, f := range features {
			bufSize += len(f) * vecSize * 2
		}
		buf := make([]float32, bufSize)
		res := make([][]kahanSum, len(features))
		for i, feats := range features {
			res[i] = make([]kahanSum, len(feats))
			for j := range res[i] {
				res[i][j] = kahanSum{
					sum:          buf[:vecSize],
					compensation: buf[vecSize : vecSize*2],
				}
				buf = buf[vecSize*2:]
			}
		}
		return res
	}

	var lock sync.Mutex
	sum := makeSums()

	numProcs := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			localSum := makeSums()
			for j := i; j < len(samples); j += numProcs {
				sample := samples[j]
				for k, horizonFeatures := range features {
					horizon := b.Horizons[k]
					if horizon > sample.Index {
						for l, f := range horizonFeatures {
							trueValue := f == -1
							if trueValue == trueIsMinority[k][l] {
								localSum[k][l].Add(sample.Vector)
							}
						}
					} else {
						ts := sample.Sequence[sample.Index-horizon]
						for l, f := range horizonFeatures {
							trueValue := f >= 0 && ts.Features.Get(f)
							if trueValue == trueIsMinority[k][l] {
								localSum[k][l].Add(sample.Vector)
							}
						}
					}
				}
			}
			lock.Lock()
			for i, x := range localSum {
				for j, s := range x {
					sum[i][j].Add(s.Sum())
				}
			}
			lock.Unlock()
		}(i)
	}
	wg.Wait()

	return sum
}

// featureSplitQuality evaluates a given split.
// The result is greater for better splits.
//
// See sortFeatures() for details on sampleFrac.
func (b *Builder) featureSplitQuality(falses, trues []vecSample, sums *lossSums, f BranchFeature,
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
		oldTrueSum[i] = x * sampleFrac
	}

	newQuality := b.Heuristic.Quality(newFalseSum) + b.Heuristic.Quality(newTrueSum)
	oldQuality := b.Heuristic.Quality(sums.False) + b.Heuristic.Quality(oldTrueSum)

	// Avoid numerically insignificant deltas.
	minDelta := math.Min(math.Abs(float64(newQuality)), math.Abs(float64(oldQuality))) * 1e-6
	if math.Abs(float64(newQuality-oldQuality)) < minDelta {
		return 0
	}

	return newQuality - oldQuality
}

func (b *Builder) evaluateFeature(samples []vecSample, f BranchFeature) (values []bool,
	falses, trues int) {
	values = make([]bool, len(samples))
	for i, s := range samples {
		val := s.BranchFeature(f)
		values[i] = val
		if val {
			trues++
		} else {
			falses++
		}
	}
	return
}

func (b *Builder) minoritySum(samples []vecSample, values []bool, trueIsMinority bool) []float32 {
	minoritySum := newKahanSum(len(samples[0].Vector))
	for i, val := range values {
		if val == trueIsMinority {
			minoritySum.Add(samples[i].Vector)
		}
	}
	return minoritySum.Sum()
}

type lossSums struct {
	False []float32
	True  []float32
}

func newLossSums(falses, trues []vecSample) *lossSums {
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

func subsampleLimit(samples []vecSample, max int) ([]vecSample, float32) {
	splitSamples := samples
	if max != 0 && len(splitSamples) > max {
		splitSamples = make([]vecSample, max)
		for i, j := range rand.Perm(len(samples))[:max] {
			splitSamples[i] = samples[j]
		}
	}
	sampleFrac := float32(float64(len(splitSamples)) / float64(len(samples)))
	return splitSamples, sampleFrac
}
