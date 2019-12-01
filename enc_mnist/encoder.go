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

	EncodingDim1    = 40
	EncodingDim2    = 20
	EncodingOptions = 8
)

func TrainEncoder(e *Encoder, ds mnist.DataSet) {
	if len(e.Layer1.Trees) < EncodingDim1 {
		trainEncoderLayer1(e, ds)
	}
	if len(e.Layer2.Trees) < EncodingDim2 {
		trainEncoderLayer2(e, ds)
	}
}

func trainEncoderLayer1(e *Encoder, ds mnist.DataSet) {
	const batch = 10000

	builder := seqtree.Builder{
		Heuristic:       seqtree.GradientHeuristic{Loss: seqtree.Sigmoid{}},
		Depth:           4,
		MinSplitSamples: 10,
		MaxUnion:        5,
		Horizons:        []int{0},
	}
	pruner := seqtree.Pruner{
		Heuristic: builder.Heuristic,
		MaxLeaves: EncodingOptions,
	}

	for e.NeedsTraining() {
		samples := generateEncodingSamples(ds, batch)
		e.Layer1.EvaluateAll(samples)

		loss := float32(0)
		for _, s := range samples {
			loss += s.MeanLoss(seqtree.Sigmoid{})
		}

		if len(e.Layer1.Trees) >= 2 {
			// First steps benefit from GradientHeuristic.
			builder.Heuristic = seqtree.PolynomialHeuristic{Loss: seqtree.Sigmoid{}}
		}
		tree := builder.Build(seqtree.TimestepSamples(samples))
		tree = pruner.Prune(seqtree.TimestepSamples(samples), tree)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			20, 1, 30)

		// Evaluate delta on a different batch.
		samples = generateEncodingSamples(ds, batch)
		e.Layer1.EvaluateAll(samples)
		delta := seqtree.AvgLossDelta(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			1.0)

		e.Layer1.Add(tree, 1.0)

		log.Printf("tree %d: loss=%f delta=%f", len(e.Layer1.Trees)-1, loss/batch, -delta)
	}
}

func trainEncoderLayer2(e *Encoder, ds mnist.DataSet) {
	const batch = 10000

	lossFunc := seqtree.MultiSoftmax{}
	for i := 0; i < EncodingDim1; i++ {
		lossFunc.Sizes = append(lossFunc.Sizes, EncodingOptions)
	}

	builder := seqtree.Builder{
		Heuristic:       seqtree.GradientHeuristic{Loss: lossFunc},
		Depth:           6,
		MinSplitSamples: 10,
		MaxUnion:        30,
		Horizons:        []int{0},
	}
	pruner := seqtree.Pruner{
		Heuristic: builder.Heuristic,
		MaxLeaves: EncodingOptions,
	}

	for e.NeedsTraining() {
		samples := generateEncodingSamplesLayer2(ds, e, batch)
		e.Layer2.EvaluateAll(samples)

		loss := float32(0)
		for _, s := range samples {
			loss += s.MeanLoss(lossFunc)
		}

		tree := builder.Build(seqtree.TimestepSamples(samples))
		tree = pruner.Prune(seqtree.TimestepSamples(samples), tree)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(samples), tree, lossFunc,
			20, 1, 30)

		// Evaluate delta on a different batch.
		samples = generateEncodingSamplesLayer2(ds, e, batch)
		e.Layer2.EvaluateAll(samples)
		delta := seqtree.AvgLossDelta(seqtree.TimestepSamples(samples), tree, lossFunc, 1.0)

		e.Layer2.Add(tree, 1.0)

		log.Printf("tree %d: loss=%f delta=%f", len(e.Layer2.Trees)-1, loss/batch, -delta)
	}
}

type Encoder struct {
	Layer1 *seqtree.Model
	Layer2 *seqtree.Model
}

func NewEncoder() *Encoder {
	return &Encoder{
		Layer1: &seqtree.Model{
			BaseFeatures: ImageSize * ImageSize,
		},
		Layer2: &seqtree.Model{
			BaseFeatures: EncodingDim1 * EncodingOptions,
		},
	}
}

func (e *Encoder) NeedsTraining() bool {
	return len(e.Layer1.Trees) < EncodingDim1 ||
		len(e.Layer2.Trees) < EncodingDim2
}

func (e *Encoder) EncodeLayer1(image []float64) []int {
	sample := sampleToTimestep(image)
	seq := seqtree.Sequence{sample}
	ts := &seqtree.TimestepSample{Sequence: seq, Index: 0}
	var res []int
	for _, t := range e.Layer1.Trees {
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

func (e *Encoder) EncodeLayer2(intSeq []int) []int {
	sample := sampleToTimestepLayer2(intSeq)
	seq := seqtree.Sequence{sample}
	ts := &seqtree.TimestepSample{Sequence: seq, Index: 0}
	var res []int
	for _, t := range e.Layer2.Trees {
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

func (e *Encoder) Encode(image []float64) []int {
	return e.EncodeLayer2(e.EncodeLayer1(image))
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

func (e *Encoder) DecodeLayer1(seq []int) []float64 {
	res := make([]float64, ImageSize*ImageSize)
	for i, leafIdx := range seq {
		leaf := e.Layer1.Trees[i].Leaves()[leafIdx]
		for j, x := range leaf.OutputDelta {
			res[j] += float64(x)
		}
	}
	return res
}

func (e *Encoder) DecodeLayer2(seq []int) []int {
	params := make([]float32, EncodingDim1*EncodingOptions)
	for i, leafIdx := range seq {
		leaf := e.Layer2.Trees[i].Leaves()[leafIdx]
		for j, x := range leaf.OutputDelta {
			params[j] += x
		}
	}
	res := make([]int, EncodingDim1)
	for i := range res {
		maxIdx := 0
		maxValue := params[i*EncodingOptions]
		for j, x := range params[i*EncodingOptions : (i+1)*EncodingOptions] {
			if x > maxValue {
				maxValue = x
				maxIdx = j
			}
		}
		res[i] = maxIdx
	}
	return res
}

func (e *Encoder) Decode(seq []int) []float64 {
	return e.DecodeLayer1(e.DecodeLayer2(seq))
}

func generateEncodingSamples(ds mnist.DataSet, n int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for _, i := range rand.Perm(len(ds.Samples))[:n] {
		sample := ds.Samples[i]
		res = append(res, seqtree.Sequence{sampleToTimestep(sample.Intensities)})
	}
	return res
}

func generateEncodingSamplesLayer2(ds mnist.DataSet, e *Encoder, n int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for _, i := range rand.Perm(len(ds.Samples))[:n] {
		sample := ds.Samples[i]
		enc1 := e.EncodeLayer1(sample.Intensities)
		res = append(res, seqtree.Sequence{sampleToTimestepLayer2(enc1)})
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

func sampleToTimestepLayer2(seq []int) *seqtree.Timestep {
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(EncodingDim1 * EncodingOptions),
		Output:   make([]float32, EncodingDim1*EncodingOptions),
		Target:   make([]float32, EncodingDim1*EncodingOptions),
	}
	for i, s := range seq {
		ts.Features.Set(i*EncodingOptions+s, true)
		ts.Target[i*EncodingOptions+s] = 1
	}
	return ts
}
