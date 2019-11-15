package seqtree

import (
	"runtime"
	"sync"
)

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
	var bestQuality float64

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

func featureSplitQuality(timesteps []*Timestep, f BranchFeature) float64 {
	vecSize := len(timesteps[0].Output)

	falseSum := make([]float64, vecSize)
	trueSum := make([]float64, vecSize)

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

	return vectorNormSquared(falseSum)/float64(falseCount) +
		vectorNormSquared(trueSum)/float64(trueCount)
}

func vectorNormSquared(v []float64) float64 {
	var res float64
	for _, x := range v {
		res += x * x
	}
	return res
}
