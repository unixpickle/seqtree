package seqtree

import (
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

// BoundedStep computes a step size that ensures that the
// update KL divergence is less than maxKL and that the
// resulting loss improves.
func BoundedStep(timesteps []*TimestepSample, t *Tree, maxKL, maxStep float32) float32 {
	for i := 0; i < 64; i++ {
		var currentKL float32
		var currentDelta float32
		for _, ts := range timesteps {
			leaf := t.Evaluate(ts)
			currentKL += SoftmaxLossKL(ts.Timestep().Output, leaf.OutputDelta, -maxStep)
			currentDelta += SoftmaxLossDelta(ts.Timestep().Output, ts.Timestep().Target,
				leaf.OutputDelta, -maxStep)
		}
		currentKL /= float32(len(timesteps))
		if currentKL <= maxKL && currentDelta < 0 {
			return maxStep
		}
		maxStep *= 0.8
	}
	return 0
}

// BuildTree builds a tree greedily for the timesteps.
//
// The minLeafSamples argument specifies the minimum
// number of samples there must be in order for the tree
// to continue attempting to split the data.
//
// The nextNewFeature argument is the number of features
// prior to adding this new tree.
func BuildTree(timesteps []*TimestepSample, depth, minLeafSamples, nextNewFeature int,
	horizons []int) *Tree {
	if len(timesteps) == 0 {
		panic("no data")
	}

	if depth == 0 || len(timesteps) <= minLeafSamples {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: gradientMean(timesteps),
				Feature:     nextNewFeature,
			},
		}
	}

	feature := OptimalFeature(timesteps, horizons)

	var falses, trues []*TimestepSample
	for _, t := range timesteps {
		if t.BranchFeature(feature) {
			trues = append(trues, t)
		} else {
			falses = append(falses, t)
		}
	}

	if len(trues) == 0 || len(falses) == 0 {
		// No split does any good.
		return BuildTree(timesteps, 0, minLeafSamples, nextNewFeature, nil)
	}

	tree1 := BuildTree(falses, depth-1, minLeafSamples, nextNewFeature, horizons)
	tree2 := BuildTree(trues, depth-1, minLeafSamples, nextNewFeature+tree1.NumFeatures(), horizons)

	return &Tree{
		Branch: &Branch{
			Feature:     feature,
			FalseBranch: tree1,
			TrueBranch:  tree2,
		},
	}
}

// OptimalFeature finds the optimal feature to split on in
// order to separate the gradients of all the timesteps.
//
// This assumes that the timesteps all have a gradient
// set.
//
// The horizons argument specifies how many timesteps in
// the past we may look. A value of zero indicates that
// only the current timestep may be inspected.
func OptimalFeature(timesteps []*TimestepSample, horizons []int) BranchFeature {
	if len(timesteps) == 0 {
		panic("no timesteps passed")
	}
	numFeatures := len(timesteps[0].Timestep().Features)
	gradSum := gradientSum(timesteps)

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
				quality := featureSplitQuality(timesteps, f, gradSum)
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
			featureChan <- BranchFeature{Feature: i, StepsInPast: horizon}
		}
	}
	close(featureChan)

	wg.Wait()

	return bestFeature
}

func featureSplitQuality(timesteps []*TimestepSample, f BranchFeature, sum []float32) float32 {
	falseCount := 0
	trueCount := 0
	featureValues := make([]bool, len(timesteps))

	for i, t := range timesteps {
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
			for j, x := range timesteps[i].Timestep().Gradient {
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
	sum := make([]float32, len(ts[0].Timestep().Gradient))
	for _, t := range ts {
		for j, x := range t.Timestep().Gradient {
			sum[j] += x
		}
	}
	return sum
}

func gradientMean(ts []*TimestepSample) []float32 {
	sum := gradientSum(ts)
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}
