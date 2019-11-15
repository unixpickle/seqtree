package seqtree

import (
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

// BuildTree builds a tree greedily for the timesteps.
//
// The nextNewFeature argument is the number of features
// prior to adding this new tree.
func BuildTree(timesteps []*Timestep, depth, nextNewFeature int, horizons []int) *Tree {
	if len(timesteps) == 0 {
		panic("no data")
	}

	if depth == 0 {
		return &Tree{
			Leaf: &Leaf{
				OutputDelta: gradientMean(timesteps),
				Feature:     nextNewFeature,
			},
		}
	}

	feature := OptimalFeature(timesteps, horizons)

	var falses, trues []*Timestep
	for _, t := range timesteps {
		if t.BranchFeature(feature) {
			trues = append(trues, t)
		} else {
			falses = append(falses, t)
		}
	}

	if len(trues) == 0 || len(falses) == 0 {
		// No split does any good.
		return BuildTree(timesteps, 0, nextNewFeature, nil)
	}

	tree1 := BuildTree(falses, depth-1, nextNewFeature, horizons)
	tree2 := BuildTree(trues, depth-1, nextNewFeature+tree1.NumFeatures(), horizons)

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
func OptimalFeature(timesteps []*Timestep, horizons []int) BranchFeature {
	if len(timesteps) == 0 {
		panic("no timesteps passed")
	}
	numFeatures := len(timesteps[0].Features)
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
		for i := 0; i < numFeatures; i++ {
			featureChan <- BranchFeature{Feature: i, StepsInPast: horizon}
		}
	}
	close(featureChan)

	wg.Wait()

	return bestFeature
}

func featureSplitQuality(timesteps []*Timestep, f BranchFeature, sum []float32) float32 {
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
			for j, x := range timesteps[i].Gradient {
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

func gradientSum(ts []*Timestep) []float32 {
	sum := make([]float32, len(ts[0].Gradient))
	for _, t := range ts {
		for j, x := range t.Gradient {
			sum[j] += x
		}
	}
	return sum
}

func gradientMean(ts []*Timestep) []float32 {
	sum := gradientSum(ts)
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}
