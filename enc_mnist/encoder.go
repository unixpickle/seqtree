package main

import (
	"log"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/seqtree"
)

const (
	ImageSize = 28

	EncodingDim     = 40
	EncodingOptions = 8
)

func TrainEncoder(e *Encoder, ds mnist.DataSet) {
	const batch = 10000

	builder := seqtree.Builder{
		Heuristic:       seqtree.GradientHeuristic{Loss: seqtree.Sigmoid{}},
		Depth:           3,
		MinSplitSamples: 10,
		MaxUnion:        5,
		Horizons:        []int{0},
	}

	for e.NeedsTraining() {
		samples := generateEncodingSamples(ds, batch)
		e.Model.EvaluateAll(samples)

		loss := float32(0)
		for _, s := range samples {
			loss += s.MeanLoss(seqtree.Sigmoid{})
		}

		if len(e.Model.Trees) != 0 {
			// First step benefits from GradientHeuristic.
			builder.Heuristic = seqtree.PolynomialHeuristic{Loss: seqtree.Sigmoid{}}
		}
		tree := builder.Build(seqtree.TimestepSamples(samples))

		samples = generateEncodingSamples(ds, batch)
		e.Model.EvaluateAll(samples)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			20, 1, 30)
		delta := seqtree.AvgLossDelta(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			1.0)
		e.Model.Add(tree, 1.0)

		log.Printf("tree %d: loss=%f delta=%f", len(e.Model.Trees)-1, loss/batch, -delta)
	}
}

type Encoder struct {
	Model *seqtree.Model
}

func NewEncoder() *Encoder {
	return &Encoder{
		Model: &seqtree.Model{
			BaseFeatures: ImageSize * ImageSize,
		},
	}
}

func (e *Encoder) NeedsTraining() bool {
	return len(e.Model.Trees) < EncodingDim
}

func (e *Encoder) Encode(image []float64) []int {
	sample := sampleToTimestep(image)
	seq := seqtree.Sequence{sample}
	ts := &seqtree.TimestepSample{Sequence: seq, Index: 0}
	var res []int
	for _, t := range e.Model.Trees {
		leaf := t.Evaluate(ts)
		var idx int
		for i, l := range t.Leaves() {
			if l == leaf {
				idx = i
				break
			}
		}
		res = append(res, idx)
	}
	return res
}

func (e *Encoder) EncodeBatch(ds mnist.DataSet, n int) [][]int {
	perm := rand.Perm(len(ds.Samples))[:n]
	res := make([][]int, n)
	numProcs := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := i; j < n; j += numProcs {
				res[j] = e.Encode(ds.Samples[perm[j]].Intensities)
			}
		}(i)
	}
	wg.Wait()
	return res
}

func (e *Encoder) Decode(seq []int) []float64 {
	res := make([]float64, ImageSize*ImageSize)
	for i, leafIdx := range seq {
		leaf := e.Model.Trees[i].Leaves()[leafIdx]
		for j, x := range leaf.OutputDelta {
			res[j] += float64(x)
		}
	}
	return res
}

func generateEncodingSamples(ds mnist.DataSet, n int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for _, i := range rand.Perm(len(ds.Samples))[:n] {
		sample := ds.Samples[i]
		res = append(res, seqtree.Sequence{sampleToTimestep(sample.Intensities)})
	}
	return res
}

func sampleToTimestep(intensities []float64) *seqtree.Timestep {
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(ImageSize * ImageSize),
		Output:   make([]float32, ImageSize*ImageSize),
		Target:   make([]float32, ImageSize*ImageSize),
	}
	for i, s := range intensities {
		if s >= 0.5 {
			ts.Features.Set(i, true)
			ts.Target[i] = 1.0
		}
	}
	return ts
}
