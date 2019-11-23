package seqtree

import (
	"runtime"
	"sync"
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
	propagateLosses(seqs, false)
}

// PropagateLossesNatural is like PropagateLosses, except
// that it uses the natural gradient.
func PropagateLossesNatural(seqs []Sequence) {
	propagateLosses(seqs, true)
}

func propagateLosses(seqs []Sequence, natural bool) {
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
