package seqtree

import (
	"runtime"
	"sync"
)

// OptimalStep performs a line search to find a step size
// that minimizes the loss.
func OptimalStep(timesteps []*TimestepSample, t *Tree, l LossFunc, maxStep float32,
	iters int) float32 {
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
						tmpOutput[i] = x + stepSize*outputDelta[i]
					}
					tmpAddition[0] = l.Loss(tmpOutput, ts.Target)
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
func ScaleOptimalStep(timesteps []*TimestepSample, t *Tree, l LossFunc, maxStep float32,
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
							tmpOutput[k] = x + stepSize*leaf.OutputDelta[k]
						}
						tmpAddition[0] = l.Loss(tmpOutput, sample.Target)
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

// scaleOptimalStepCluster is like ScaleOptimalStep, but
// for a single delta applied to an entire cluster of
// data.
//
// This can be used to only scale steps within a given
// range, allowing for efficient steps with MultiSoftmax
// and other block-diagonal loss functions.
func scaleOptimalStepCluster(data, targets, otherData, otherTargets [][]float32, delta []float32,
	l LossFunc, maxStep, otherWeight float32, iters, startIdx, length int) {
	if length == 0 {
		length = len(delta) - startIdx
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
				tmpOutput := make([]float32, length)
				for j := i; j < len(data); j += numProcs {
					sample := data[j][startIdx : startIdx+length]
					target := targets[j][startIdx : startIdx+length]
					for k, x := range sample {
						tmpOutput[k] = x + stepSize*delta[k+startIdx]
					}
					tmpAddition[0] = l.Loss(tmpOutput, target)
					total.Add(tmpAddition)
				}
				for j := i; j < len(otherData); j += numProcs {
					sample := otherData[j][startIdx : startIdx+length]
					target := otherTargets[j][startIdx : startIdx+length]
					for k, x := range sample {
						tmpOutput[k] = x + stepSize*delta[k+startIdx]
					}
					tmpAddition[0] = -otherWeight * l.Loss(tmpOutput, target)
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
	for i := startIdx; i < startIdx+length; i++ {
		delta[i] *= scale
	}
}

// AvgLossDelta computes the average change in the loss
// after taking a step.
func AvgLossDelta(timesteps []*TimestepSample, t *Tree, l LossFunc, step float32) float32 {
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
				oldLoss := l.Loss(ts.Timestep().Output, ts.Timestep().Target)
				newOut := addDelta(ts.Timestep().Output, leaf.OutputDelta, step)
				newLoss := l.Loss(newOut, ts.Timestep().Target)
				deltaTotal.Add([]float32{newLoss - oldLoss})
			}
			lock.Lock()
			currentDelta += deltaTotal.Sum()[0]
			lock.Unlock()
		}(i)
	}
	wg.Wait()
	return currentDelta / float32(len(timesteps))
}
