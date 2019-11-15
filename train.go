package seqtree

import (
	"runtime"
	"sync"
)

// BuildTree builds a tree greedily for the timesteps.
//
// The nextNewFeature argument is the number of features
// prior to adding this new tree.
func BuildTree(timesteps []*Timestep, horizon, depth, nextNewFeature int) *Tree {
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

	feature := OptimalFeature(timesteps, horizon)

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
		return BuildTree(timesteps, 0, 0, nextNewFeature)
	}

	tree1 := BuildTree(falses, horizon, depth-1, nextNewFeature)
	tree2 := BuildTree(trues, horizon, depth-1, nextNewFeature+tree1.NumFeatures())

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
// The horizon argument specifies how many timesteps in
// the past we may look. A value of zero indicates that
// only the current timestep may be inspected.
func OptimalFeature(timesteps []*Timestep, horizon int) BranchFeature {
	if len(timesteps) == 0 {
		panic("no timesteps passed")
	}
	numFeatures := len(timesteps[0].Features)

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
				quality := featureSplitQuality(timesteps, f)
				resultLock.Lock()
				if quality >= bestQuality {
					bestFeature = f
					bestQuality = quality
				}
				resultLock.Unlock()
			}
		}()
	}

	for i := 0; i <= horizon; i++ {
		for j := 0; j < numFeatures; j++ {
			featureChan <- BranchFeature{Feature: j, StepsInPast: i}
		}
	}
	close(featureChan)

	wg.Wait()

	return bestFeature
}

func featureSplitQuality(timesteps []*Timestep, f BranchFeature) float32 {
	vecSize := len(timesteps[0].Output)

	falseSum := make([]float32, vecSize)
	trueSum := make([]float32, vecSize)

	falseCount := 0
	trueCount := 0

	for _, t := range timesteps {
		arr := falseSum
		if t.BranchFeature(f) {
			arr = trueSum
			trueCount++
		} else {
			falseCount++
		}
		for i, x := range t.Gradient {
			arr[i] += x
		}
	}

	if falseCount == 0 || trueCount == 0 {
		return 0
	}

	return vectorNormSquared(falseSum)/float32(falseCount) +
		vectorNormSquared(trueSum)/float32(trueCount)
}

func vectorNormSquared(v []float32) float32 {
	var res float32
	for _, x := range v {
		res += x * x
	}
	return res
}

func gradientMean(ts []*Timestep) []float32 {
	sum := make([]float32, len(ts[0].Gradient))
	for _, t := range ts {
		for j, x := range t.Gradient {
			sum[j] += x
		}
	}
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}
